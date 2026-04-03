"""
Model definitions for Q2: ResNet-18 (CIFAR-10 classification) and
ResNet-34 (adversarial detection).
Both modified for 32x32 input images.
"""

import torch
import torch.nn as nn
from torchvision import models


class CIFARResNet18(nn.Module):
    """
    ResNet-18 modified for CIFAR-10 (32x32 images).
    Changes from standard ResNet:
    - First conv: 3x3, stride=1, padding=1 (instead of 7x7, stride=2)
    - No max pooling after first conv
    """

    def __init__(self, num_classes=10, pretrained=False):
        super().__init__()
        base = models.resnet18(pretrained=pretrained)

        # Modify first layer for 32x32 images
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu
        # Skip maxpool for 32x32 images

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.avgpool = base.avgpool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class CIFARResNet34(nn.Module):
    """
    ResNet-34 modified for CIFAR images.
    Used as binary classifier for adversarial detection.
    """

    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        base = models.resnet34(pretrained=pretrained)

        # Modify first layer for 32x32 images
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.avgpool = base.avgpool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def create_resnet18(num_classes=10):
    """Create a CIFAR-10 ResNet-18 (non-pretrained)."""
    model = CIFARResNet18(num_classes=num_classes, pretrained=False)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Model] CIFAR-ResNet18 created. Parameters: {total:,}")
    return model


def create_resnet34(num_classes=2):
    """Create a CIFAR ResNet-34 for binary adversarial detection."""
    model = CIFARResNet34(num_classes=num_classes, pretrained=False)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Model] CIFAR-ResNet34 (detector) created. Parameters: {total:,}")
    return model


if __name__ == "__main__":
    # Test models
    model18 = create_resnet18()
    x = torch.randn(2, 3, 32, 32)
    out = model18(x)
    print(f"ResNet-18 output: {out.shape}")

    model34 = create_resnet34()
    out = model34(x)
    print(f"ResNet-34 output: {out.shape}")
