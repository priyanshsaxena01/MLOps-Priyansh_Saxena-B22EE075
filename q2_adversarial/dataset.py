"""
CIFAR-10 Dataset loading for Q2 adversarial attacks.
Data is normalized to [0, 1] range for ART compatibility.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_cifar10_transforms():
    """Get train and test transforms for CIFAR-10 (ResNet)."""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),  # Scales to [0, 1]
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),  # Scales to [0, 1]
    ])

    return train_transform, test_transform


def get_cifar10_dataloaders(batch_size=128, num_workers=4, val_split=5000,
                             data_dir="./data"):
    """
    Create CIFAR-10 train, validation, and test dataloaders.

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    train_transform, test_transform = get_cifar10_transforms()

    full_train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    full_val_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=test_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    # Split
    train_size = len(full_train_dataset) - val_split
    train_indices, val_indices = random_split(
        range(len(full_train_dataset)),
        [train_size, val_split],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        full_train_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        full_val_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    class_names = test_dataset.classes
    print(f"[CIFAR-10] Train: {train_size}, Val: {val_split}, "
          f"Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, class_names


def get_cifar10_numpy(data_dir="./data"):
    """
    Get CIFAR-10 as numpy arrays for ART.
    Returns arrays in [0, 1] range with shape (N, 3, 32, 32) for PyTorch.

    Returns:
        x_train, y_train, x_test, y_test
    """
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=transforms.ToTensor()
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True,
        transform=transforms.ToTensor()
    )

    # Convert to numpy
    x_train = np.array([img.numpy() for img, _ in train_dataset]).astype(np.float32)
    y_train = np.array([label for _, label in train_dataset])

    x_test = np.array([img.numpy() for img, _ in test_dataset]).astype(np.float32)
    y_test = np.array([label for _, label in test_dataset])

    print(f"[CIFAR-10 NumPy] Train: {x_train.shape}, Test: {x_test.shape}")
    print(f"[CIFAR-10 NumPy] Value range: [{x_train.min():.3f}, {x_train.max():.3f}]")

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_cifar10_dataloaders()
    print(f"Classes: {classes}")

    images, labels = next(iter(train_loader))
    print(f"Batch: {images.shape}, Range: [{images.min():.3f}, {images.max():.3f}]")

    x_train, y_train, x_test, y_test = get_cifar10_numpy()
