import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torchvision.models import resnet50, ResNet50_Weights
from data.dataset import ImageNetDataModule
import signal
import sys
import yaml
from pathlib import Path
from datetime import datetime, timedelta

# Set matmul precision
torch.set_float32_matmul_precision('high')

class ImageNetModule(pl.LightningModule):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Optionally freeze early layers
        for param in list(self.model.parameters())[:-3]:
            param.requires_grad = False
            
        self.criterion = nn.CrossEntropyLoss()
        self.validation_step_outputs = []
        self.save_hyperparameters()
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('train_acc', acc, prog_bar=True, sync_dist=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc1 = (logits.argmax(dim=1) == y).float().mean()
        _, pred = logits.topk(5, 1, True, True)
        acc5 = (pred == y.view(-1, 1)).float().max(dim=1)[0].mean()
        
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc_top1', acc1, prog_bar=True, sync_dist=True)
        self.log('val_acc_top5', acc5, prog_bar=True, sync_dist=True)
        
        # Store outputs for epoch end
        self.validation_step_outputs.append({
            'val_loss': loss,
            'val_acc_top1': acc1,
            'val_acc_top5': acc5
        })
        
    def on_validation_epoch_end(self):
        """Log epoch-level validation metrics."""
        outputs = self.validation_step_outputs
        
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc1 = torch.stack([x['val_acc_top1'] for x in outputs]).mean()
        avg_acc5 = torch.stack([x['val_acc_top5'] for x in outputs]).mean()
        
        # Log metrics in a cleaner format
        metrics = {
            'epoch': self.current_epoch,
            'val_loss': f"{avg_loss:.4f}",
            'val_acc_top1': f"{avg_acc1:.4f}",
            'val_acc_top5': f"{avg_acc5:.4f}"
        }
        
        # Print in a cleaner format
        self.print("\n" + "="*50)
        self.print(f"Validation Results - Epoch {self.current_epoch}")
        self.print("-"*50)
        self.print(f"Loss:          {metrics['val_loss']}")
        self.print(f"Top-1 Accuracy: {metrics['val_acc_top1']}")
        self.print(f"Top-5 Accuracy: {metrics['val_acc_top5']}")
        self.print("="*50 + "\n")
        
        # Log for tensorboard/CSV
        self.log('epoch_val_loss', avg_loss, prog_bar=True, sync_dist=True)
        self.log('epoch_val_acc_top1', avg_acc1, prog_bar=True, sync_dist=True)
        self.log('epoch_val_acc_top5', avg_acc5, prog_bar=True, sync_dist=True)
        
        # Clear the outputs list
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        # Use gradient accumulation for larger effective batch size
        # This allows using smaller, cheaper instances
        params = [
            {'params': [p for p in self.model.parameters() if not p.requires_grad], 'lr': 1e-4},
            {'params': [p for p in self.model.parameters() if p.requires_grad], 'lr': 1e-3}
        ]
        
        optimizer = torch.optim.AdamW(params, weight_decay=0.05)
        
        # Cosine schedule with warmup for better convergence
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[1e-4, 1e-3],
                epochs=self.trainer.max_epochs,
                steps_per_epoch=1000,
                pct_start=0.1,
                div_factor=10,
                final_div_factor=100,
            ),
            'interval': 'step'
        }
        
        return [optimizer], [scheduler]

class SpotTerminationHandler:
    def __init__(self, trainer):
        self.trainer = trainer
        self.terminating = False
        signal.signal(signal.SIGTERM, self.handle_sigterm)

    def handle_sigterm(self, signum, frame):
        if not self.terminating:
            print("Received termination signal. Saving checkpoint...")
            self.terminating = True
            self.trainer.save_checkpoint("checkpoints/spot_interrupted.ckpt")
            sys.exit(0)

class CustomProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        # Remove version number
        items.pop("v_num", None)
        # Keep only essential metrics
        essential_metrics = {
            'loss': items.get('train_loss', 0.0),
            'acc': items.get('train_acc', 0.0),
            'val_acc': items.get('val_acc_top1', 0.0)
        }
        return essential_metrics

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description("Validating")
        return bar

def load_config():
    config_path = Path("configs/config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    pl.seed_everything(42)
    
    # Create logger with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = CSVLogger(
        save_dir='logs',
        name=f'training_{timestamp}',
        flush_logs_every_n_steps=config['monitoring']['log_every_n_steps']
    )
    
    # Load checkpoint if exists
    ckpt_path = None
    latest_ckpt = Path("checkpoints/last.ckpt")
    if latest_ckpt.exists():
        ckpt_path = str(latest_ckpt)
        print(f"Resuming from checkpoint: {ckpt_path}")
    
    data_module = ImageNetDataModule(
        train_dir=config['data']['train_dir'],
        val_dir=config['data']['val_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        persistent_workers=config['training']['persistent_workers'],
        prefetch_factor=config['data']['prefetch_factor'],
        image_size=config['data']['image_size'],
        random_erase_prob=config['data']['random_erase_prob']
    )
    
    model = ImageNetModule()
    
    callbacks = [
        ModelCheckpoint(
            dirpath='checkpoints',
            filename='resnet50-{epoch:02d}-{epoch_val_acc_top1:.4f}',
            monitor='epoch_val_acc_top1',
            mode='max',
            save_top_k=3,
            save_last=True
        ),
        LearningRateMonitor(logging_interval='step'),
        CustomProgressBar(refresh_rate=50)
    ]
    
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=4,
        strategy=DDPStrategy(
            process_group_backend="nccl",
            find_unused_parameters=False,
            timeout=timedelta(seconds=1800),
            broadcast_buffers=False,
            static_graph=True,
            gradient_as_bucket_view=True,
            ddp_comm_hook=None,
            ddp_comm_state=None,
        ),
        precision='16-mixed',
        max_epochs=30,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=50,
        val_check_interval=0.5,
        num_sanity_val_steps=0
    )
    
    # Initialize spot termination handler
    spot_handler = SpotTerminationHandler(trainer)
    
    # Use fit with ckpt_path instead of resume_from_checkpoint
    trainer.fit(model, data_module, ckpt_path=ckpt_path)

if __name__ == '__main__':
    main() 