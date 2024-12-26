import os
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
import numpy as np
import torch.cuda.nvtx as nvtx

from models.resnet import create_resnet50
from data.dataset import create_dataloaders
from utils.metrics import AverageMeter, accuracy
from utils.logger import setup_logger
from utils.mixup import mixup_data, cutmix_data

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_epoch(epoch, model, train_loader, criterion, optimizer, scaler, 
                device, config, logger):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    optimizer.zero_grad(set_to_none=True)
    
    num_steps = len(train_loader)
    accum_steps = config['training']['gradient_accumulation_steps']
    
    pbar = tqdm(train_loader)
    
    for step, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        
        r = np.random.rand(1)
        if r < 0.5 and config['training']['mixup_alpha'] > 0:
            images, targets_a, targets_b, lam = mixup_data(
                images, targets, config['training']['mixup_alpha'])
        elif config['training']['cutmix_alpha'] > 0:
            images, targets_a, targets_b, lam = cutmix_data(
                images, targets, config['training']['cutmix_alpha'])
        else:
            targets_a = targets_b = targets
            lam = 1
        
        with autocast(enabled=config['training']['amp']):
            outputs = model(images)
            loss = (lam * criterion(outputs, targets_a) + 
                   (1 - lam) * criterion(outputs, targets_b)) / accum_steps
        
        scaler.scale(loss).backward()
        
        if (step + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item() * accum_steps, images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        
        pbar.set_description(
            f'Epoch [{epoch}/{config["training"]["num_epochs"]}] '
            f'Loss: {losses.avg:.4f} '
            f'Top1: {top1.avg:.2f}% '
            f'Top5: {top5.avg:.2f}%'
        )
    
    return losses.avg, top1.avg, top5.avg

def validate(model, val_loader, criterion, device, config):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Validation'):
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
    
    return losses.avg, top1.avg, top5.avg

def main(rank, world_size, config):
    setup(rank, world_size)
    
    torch.backends.cudnn.benchmark = True
    if config['training']['channels_last']:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format
    
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    model = create_resnet50(config['model']['num_classes'])
    model = model.to(device, memory_format=memory_format)
    model = DistributedDataParallel(model, 
                device_ids=[rank],
                find_unused_parameters=config['distributed']['find_unused_parameters'])
    
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config['training']['label_smoothing']
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['training']['learning_rate'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['training']['learning_rate'],
        epochs=config['training']['num_epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=config['training']['warmup_epochs'] / config['training']['num_epochs'],
        anneal_strategy='cos'
    )
    
    scaler = GradScaler()
    
    train_loader, val_loader = create_dataloaders(config, world_size, rank)
    
    logger = setup_logger(rank)
    
    if rank == 0:
        wandb.init(project="imagenet-resnet50", config=config)
    
    for epoch in range(config['training']['num_epochs']):
        train_loss, train_acc1, train_acc5 = train_epoch(
            epoch, model, train_loader, criterion, optimizer, scaler,
            device, config, logger
        )
        
        val_loss, val_acc1, val_acc5 = validate(
            model, val_loader, criterion, device, config
        )
        
        scheduler.step()
        
        if rank == 0:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc1': train_acc1,
                'train_acc5': train_acc5,
                'val_loss': val_loss,
                'val_acc1': val_acc1,
                'val_acc5': val_acc5,
                'learning_rate': scheduler.get_last_lr()[0]
            })
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_acc1': train_acc1,
                'val_acc1': val_acc1,
            }, f'checkpoints/epoch_{epoch}.pth')
    
    cleanup()

if __name__ == '__main__':
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    world_size = torch.cuda.device_count()
    mp.spawn(
        main,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    ) 