import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def create_resnet50(num_classes=1000, pretrained=False):
    if pretrained:
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        
        # Freeze early layers
        for name, param in model.named_parameters():
            if 'layer1' in name or 'layer2' in name:
                param.requires_grad = False
    else:
        model = resnet50(weights=None)
        
        # Initialize weights if not pretrained
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    return model 