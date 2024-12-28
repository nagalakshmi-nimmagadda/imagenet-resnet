import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms, datasets
from torch.utils.data.distributed import DistributedSampler
import torch
import logging

class ImageNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        batch_size: int = 256,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        image_size: int = 224,
        random_erase_prob: float = 0.2
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.image_size = image_size
        
        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Enhanced training augmentations
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            self.normalize,
            transforms.RandomErasing(p=random_erase_prob)
        ])
        
        # Validation transform
        self.val_transform = transforms.Compose([
            transforms.Resize(int(image_size * 256/224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            self.normalize,
        ])
        
        self.logger = logging.getLogger(__name__)

    def prepare_data(self):
        """Verify data exists and log dataset info."""
        if not os.path.exists(self.train_dir):
            raise RuntimeError(f"Training directory not found: {self.train_dir}")
        if not os.path.exists(self.val_dir):
            raise RuntimeError(f"Validation directory not found: {self.val_dir}")
            
        train_classes = len(os.listdir(self.train_dir))
        val_classes = len(os.listdir(self.val_dir))
        
        self.logger.info(f"Found {train_classes} training classes")
        self.logger.info(f"Found {val_classes} validation classes")

    def setup(self, stage=None):
        """Set up datasets with error handling and logging."""
        try:
            self.train_dataset = datasets.ImageFolder(
                self.train_dir,
                transform=self.train_transform
            )
            self.logger.info(f"Training dataset size: {len(self.train_dataset)}")
            
            self.val_dataset = datasets.ImageFolder(
                self.val_dir,
                transform=self.val_transform
            )
            self.logger.info(f"Validation dataset size: {len(self.val_dataset)}")
            
        except Exception as e:
            self.logger.error(f"Error setting up datasets: {str(e)}")
            raise

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Disable shuffle as DistributedSampler will handle it
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            sampler=DistributedSampler(self.train_dataset, shuffle=True)
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            sampler=DistributedSampler(self.val_dataset, shuffle=False)
        )
        
    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Monitor GPU memory usage after batch transfer."""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1e9  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / 1e9    # GB
                self.logger.debug(
                    f"GPU {i} Memory: {memory_allocated:.2f}GB allocated, "
                    f"{memory_reserved:.2f}GB reserved"
                )
        return batch 

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
        
        return {
            'val_loss': loss,
            'val_acc_top1': acc1,
            'val_acc_top5': acc5
        } 