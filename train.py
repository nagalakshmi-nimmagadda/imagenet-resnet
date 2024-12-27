import os
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.amp import autocast, GradScaler
import torch.cuda.amp
from tqdm import tqdm
import logging

from models.resnet import create_resnet50
from data.dataset import create_dataloaders
from utils.metrics import AverageMeter, accuracy
from utils.logger import setup_logger
from utils.mixup import mixup_data, cutmix_data
from utils.early_stopping import EarlyStopping
from utils.monitoring import CostMonitor
from torch.optim.lr_scheduler import OneCycleLR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_config(config):
    """Validate and convert config values to correct types."""
    try:
        config['training']['learning_rate'] = float(config['training']['learning_rate'])
        config['training']['momentum'] = float(config['training']['momentum'])
        config['training']['weight_decay'] = float(config['training']['weight_decay'])
        config['training']['label_smoothing'] = float(config['training']['label_smoothing'])
        config['training']['mixup_alpha'] = float(config['training']['mixup_alpha'])
        config['training']['cutmix_alpha'] = float(config['training']['cutmix_alpha'])
        config['monitoring']['cost_per_hour'] = float(config['monitoring']['cost_per_hour'])
        config['monitoring']['max_budget_usd'] = float(config['monitoring']['max_budget_usd'])
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid config value: {str(e)}")
    return config

def train_epoch(epoch, model, train_loader, criterion, optimizer, scaler, scheduler, device, config):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    optimizer.zero_grad(set_to_none=True)
    
    pbar = tqdm(train_loader, desc=f'Epoch [{epoch}]')
    for i, (images, target) in enumerate(pbar):
        # Move to GPU
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # Apply mixup or cutmix with probability
        r = torch.rand(1)
        if r < 0.5 and config['training']['mixup_alpha'] > 0:
            images, targets_a, targets_b, lam = mixup_data(
                images, target, config['training']['mixup_alpha'])
        elif r < 0.8 and config['training']['cutmix_alpha'] > 0:
            images, targets_a, targets_b, lam = cutmix_data(
                images, target, config['training']['cutmix_alpha'])
        else:
            targets_a = targets_b = target
            lam = 1
            
        # Forward pass with autocast
        with autocast(device_type='cuda', enabled=config['training']['amp']):
            output = model(images)
            loss = lam * criterion(output, targets_a) + (1 - lam) * criterion(output, targets_b)
            
        # Backward pass with gradient accumulation
        loss = loss / config['training']['gradient_accumulation_steps']
        scaler.scale(loss).backward()
        
        if (i + 1) % config['training']['gradient_accumulation_steps'] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_val'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
        # Metrics
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item() * config['training']['gradient_accumulation_steps'], images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc@1': f'{top1.avg:.2f}%',
            'Acc@5': f'{top5.avg:.2f}%',
            'LR': f'{current_lr:.4f}'
        })
        
    return losses.avg, top1.avg, top5.avg

def validate(model, val_loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        for images, target in tqdm(val_loader, desc='Validate'):
            images = images.to(device)
            target = target.to(device)
            
            output = model(images)
            loss = criterion(output, target)
            
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
    return losses.avg, top1.avg, top5.avg

def main():
    # Initialize process group
    dist.init_process_group(backend="nccl")
    
    # Get rank and world size
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Load and validate config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config = validate_config(config)
    
    # Setup device
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # Create model
    model = create_resnet50(config['model']['num_classes'])
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[rank])
    
    # Setup training
    criterion = nn.CrossEntropyLoss(label_smoothing=config['training']['label_smoothing'])
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        momentum=float(config['training']['momentum']),
        weight_decay=float(config['training']['weight_decay'])
    )
    
    scaler = GradScaler('cuda')
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'],
        min_delta=config['training']['early_stopping_delta']
    )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config, world_size, rank)
    
    # Setup monitoring
    if rank == 0:
        logger = setup_logger(rank)
        cost_monitor = CostMonitor(
            config['monitoring']['instance_type'],
            config['monitoring']['cost_per_hour']
        )
    
    # Set memory format
    if config['training']['channels_last']:
        model = model.to(memory_format=torch.channels_last)
    
    # Enable cudnn benchmarking
    torch.backends.cudnn.benchmark = True
    
    # Create scheduler with aggressive warmup and cosine decay
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['training']['learning_rate'],
        epochs=config['training']['num_epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.05,  # 5% warmup
        div_factor=10,   # Initial lr = max_lr/10
        final_div_factor=1e3,  # Min lr = initial_lr/1000
        anneal_strategy='cos',
        three_phase=True  # Use three-phase schedule
    )
    
    # Training loop
    for epoch in range(config['training']['num_epochs']):
        train_loss, train_acc1, train_acc5 = train_epoch(
            epoch, model, train_loader, criterion, optimizer, scaler, scheduler, device, config
        )
        
        val_loss, val_acc1, val_acc5 = validate(model, val_loader, criterion, device)
        
        if rank == 0:
            metrics = cost_monitor.log_metrics()
            logger.info(
                f"Epoch {epoch} - Train Loss: {train_loss:.4f}, "
                f"Train Acc@1: {train_acc1:.2f}%, Val Loss: {val_loss:.4f}, "
                f"Val Acc@1: {val_acc1:.2f}%, Cost: ${metrics['cost_usd']:.2f}"
            )
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc1': train_acc1,
                'val_acc1': val_acc1
            }, f'checkpoints/epoch_{epoch}.pth')
            
            # Early stopping
            early_stopping(val_loss, model, epoch, optimizer, 'checkpoints/best_model.pth')
            if early_stopping.early_stop:
                logger.info("Early stopping triggered!")
                break
            
            # Check budget
            if metrics['cost_usd'] > config['monitoring']['max_budget_usd']:
                logger.info("Budget exceeded! Stopping training...")
                break
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main() 