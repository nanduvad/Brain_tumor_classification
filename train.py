# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from Preprocess import load_data, visualize_data
from model import CNNClassifier_regularization

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)              # logits (batch, num_classes)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += los
            s.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, 100.0 * correct / total

def main():
    # Config
    data_dir = "brain_tumor"
    batch_size = 16
    epochs = 5
    lr = 0.001
    weight_decay = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "best_model.pth"

    # Load data
    train_loader, val_loader, classes, full_dataset = load_data(data_dir, batch_size=batch_size, augment=True)
    print("Classes (order used by ImageFolder):", classes)

    # Debug: visualize a few samples (optional)
    # visualize_data(train_loader, classes)

    num_classes = len(classes)
    model = CNNClassifier_regularization(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{epochs} | train_loss: {train_loss:.4f} train_acc: {train_acc:.2f}% | val_loss: {val_loss:.4f} val_acc: {val_acc:.2f}% | time: {time.time()-t0:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "state_dict": model.state_dict(),
                "classes": classes,
                "epoch": epoch,
                "val_acc": val_acc
            }
            torch.save(checkpoint, save_path)
            print(f"Saved improved model to {save_path} (val_acc={val_acc:.2f}%)")

    total_time = time.time() - start_time
    print(f"Training complete in {total_time/60:.2f} minutes. Best val acc: {best_val_acc:.2f}%")
    print(f"Best model saved at: {save_path}")

if __name__ == "__main__":
    main()
