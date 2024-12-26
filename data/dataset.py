import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import autoaugment

class ImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, cache_mode="part"):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.cache_mode = cache_mode
        self.cache = {}
        
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.samples.append((
                    os.path.join(class_dir, img_name),
                    self.class_to_idx[class_name]
                ))

        # Cache frequently used images if cache_mode is 'part'
        if cache_mode == "part":
            num_cache = len(self.samples) // 10  # Cache 10% of dataset
            for idx in range(num_cache):
                img_path, _ = self.samples[idx]
                self.cache[img_path] = Image.open(img_path).convert('RGB')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Try to get image from cache
        if img_path in self.cache:
            image = self.cache[img_path]
        else:
            image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_dataloaders(config, world_size, rank):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['data']['image_size']),
        transforms.RandomHorizontalFlip(),
        autoaugment.AutoAugment() if config['data']['auto_augment'] else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['data']['mean'], std=config['data']['std']),
        transforms.RandomErasing(p=config['data']['random_erase_prob'])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config['data']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['data']['mean'], 
                           std=config['data']['std'])
    ])

    train_dataset = ImageNetDataset(
        root_dir=config['data']['train_dir'],
        transform=train_transform,
        train=True
    )

    val_dataset = ImageNetDataset(
        root_dir=config['data']['val_dir'],
        transform=val_transform,
        train=False
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        sampler=val_sampler,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    return train_loader, val_loader 