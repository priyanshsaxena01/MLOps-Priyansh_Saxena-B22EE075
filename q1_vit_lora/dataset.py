"""
CIFAR-100 Dataset loading and preprocessing for ViT-S fine-tuning.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_cifar100_transforms(image_size=224):
    """Get train and test transforms for CIFAR-100 with ViT-S."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(image_size, padding=16),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        ),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, test_transform


def get_cifar100_dataloaders(batch_size=32, num_workers=4, val_split=5000,
                              image_size=224, data_dir="./data"):
    """
    Create CIFAR-100 train, validation, and test dataloaders.

    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        val_split: Number of samples to use for validation
        image_size: Input image size (224 for ViT)
        data_dir: Directory to download/store data

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    train_transform, test_transform = get_cifar100_transforms(image_size)

    # Download datasets
    full_train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )

    # For validation, we use test transform (no augmentation)
    full_val_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=test_transform
    )

    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    # Split train into train and validation
    train_size = len(full_train_dataset) - val_split
    train_indices, val_indices = random_split(
        range(len(full_train_dataset)),
        [train_size, val_split],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_val_dataset, val_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset.dataset if hasattr(val_dataset, 'dataset') else val_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    class_names = full_train_dataset.classes

    print(f"[Dataset] Train: {train_size}, Val: {val_split}, Test: {len(test_dataset)}")
    print(f"[Dataset] Classes: {len(class_names)}, Batch size: {batch_size}")

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = get_cifar100_dataloaders()
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Verify a batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
