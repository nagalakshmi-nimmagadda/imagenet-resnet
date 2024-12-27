import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

class ImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, rank=0):
        self.root_dir = root_dir
        self.transform = transform
        self.rank = rank
        
        if not os.path.exists(root_dir):
            raise RuntimeError(f"Dataset directory {root_dir} does not exist!")
            
        self.classes = sorted([d for d in os.listdir(root_dir) 
                             if os.path.isdir(os.path.join(root_dir, d))])
        
        if len(self.classes) == 0:
            raise RuntimeError(f"No class directories found in {root_dir}")
            
        if self.rank == 0:
            print(f"Found {len(self.classes)} classes in {root_dir}")
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                    self.samples.append((
                        os.path.join(class_dir, img_name),
                        self.class_to_idx[class_name]
                    ))
        
        if self.rank == 0:
            print(f"Found {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_dataloaders(config, world_size, rank):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['data']['image_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.RandomErasing(p=config['data']['random_erase_prob']),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['data']['mean'], std=config['data']['std'])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config['data']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['data']['mean'], std=config['data']['std'])
    ])

    train_dataset = ImageNetDataset(
        root_dir=config['data']['train_dir'],
        transform=train_transform,
        rank=rank
    )

    val_dataset = ImageNetDataset(
        root_dir=config['data']['val_dir'],
        transform=val_transform,
        rank=rank
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

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