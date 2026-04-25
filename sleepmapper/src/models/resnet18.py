import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class SleepResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super(SleepResNet18, self).__init__()
        
        # Load pretrained ResNet-18
        if pretrained:
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            self.model = resnet18(weights=None)
            
        # Modify the first convolutional layer to accept 1-channel input instead of 3
        # Original conv1: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        original_conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        
        # If pretrained, initialize the new conv1 weights by averaging the original weights across channels
        if pretrained:
            with torch.no_grad():
                self.model.conv1.weight.copy_(original_conv1.weight.mean(dim=1, keepdim=True))
                
        # Replace the final fully connected layer for binary classification
        num_ftrs = self.model.fc.in_features
        
        # Add Dropout(0.3) before final layer
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, 1) # Single logit for binary classification
        )

    def forward(self, x):
        # Forward pass returning logits
        logits = self.model(x)
        return logits
