import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from model import SimpleCNN
from utils import save_checkpoint, evaluate
from tqdm import tqdm
import os
import numpy as np  # ✅ missing import

# Config
DATA_DIR = "data"  # adjust if needed
BATCH_SIZE = 16
EPOCHS = 12
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 224

# Transforms
train_tfms = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
val_tfms = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ✅ Safe dataset loading
try:
    full_ds = datasets.ImageFolder(DATA_DIR, transform=train_tfms)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Split dataset
n = len(full_ds)
val_size = int(0.2 * n)
train_size = n - val_size
train_ds, val_ds = random_split(full_ds, [train_size, val_size])
val_ds.dataset.transform = val_tfms

# ✅ Fix for Windows: set num_workers=0 to avoid worker crash
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# Model setup
model = SimpleCNN(num_classes=len(full_ds.classes)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for xb, yb in pbar:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
        pbar.set_postfix(loss=running_loss/total, acc=correct/total)

    preds, targets = evaluate(model, val_loader, DEVICE)
    val_acc = (np.array(preds) == np.array(targets)).mean()
    print(f"Validation accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_checkpoint({
            'model_state': model.state_dict(),
            'opt_state': optimizer.state_dict(),
            'epoch': epoch,
            'classes': full_ds.classes
        }, fname="best_checkpoint.pth")
        print("Saved best model.")

print("Training finished. Best val acc:", best_val_acc)
