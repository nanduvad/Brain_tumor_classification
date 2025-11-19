# Preprocess.py
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Training transforms (augmentation)
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Validation / inference transforms
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_data(data_dir, batch_size=32, augment=True, val_split=0.3, seed=42):
    """
    Returns: train_loader, val_loader, class_names, full_dataset
    """
    transform = TRAIN_TRANSFORM if augment else VAL_TRANSFORM

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    # Important: dataset.classes gives the class order (alphabetical by default unless folders differ)
    class_names = dataset.classes
    # Split
    total = len(dataset)
    val_size = int(val_split * total)
    train_size = total - val_size

    torch.manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # For validation, ensure we use VAL_TRANSFORM (no augment)
    val_dataset.dataset = datasets.ImageFolder(root=data_dir, transform=VAL_TRANSFORM)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, class_names, dataset

def visualize_data(train_loader, classes, num_samples=5):
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    images = images.cpu().numpy().transpose(0, 2, 3, 1)
    # Denormalize from [-1,1] to [0,1]
    images = (images * 0.5) + 0.5
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        axes[i].imshow(np.clip(images[i], 0, 1))
        axes[i].axis('off')
        axes[i].set_title(classes[labels[i].item()])
    plt.show()
