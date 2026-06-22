import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

ROOT       = r"C:\Users\Your_Path\AssistNet\AssistNet_v2"
TRAIN_DIR  = os.path.join(ROOT, "data", "train")
VAL_DIR    = os.path.join(ROOT, "data", "val")
TEST_DIR   = os.path.join(ROOT, "data", "test")
MODELS_DIR = os.path.join(ROOT, "models")
PLOTS_DIR  = os.path.join(ROOT, "plots")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

IMG_SIZE   = (150, 150)
BATCH_SIZE = 64
EPOCHS     = 20
LR         = 1e-3

#device info
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


train_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# dataset(s)
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset   = datasets.ImageFolder(VAL_DIR,   transform=eval_transforms)
test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=eval_transforms)

print(f"\nClass mapping: {train_dataset.class_to_idx}")
print(f"Train: {len(train_dataset)} images")
print(f"Val:   {len(val_dataset)} images")
print(f"Test:  {len(test_dataset)} images\n")


class_counts = np.bincount(train_dataset.targets)
n_runway  = class_counts[train_dataset.class_to_idx['runway']]
n_taxiway = class_counts[train_dataset.class_to_idx['taxiway']]

print(f"Class counts — runway: {n_runway}  taxiway: {n_taxiway}\n")


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)


# MODEL AssistNet v2.0

# changes: removed AdaptiveAvgPool2d for ONNX/verifier compatibility
# sigmoid activation removed
# feature maps are 7x7 after 4 MaxPool layers on 150x150 input
# so we flatten directly to 64*7*7 = 3136
class AssistNetV2(nn.Module):
    def __init__(self):
        super(AssistNetV2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = AssistNetV2().to(device)
print(model)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTrainable parameters: {total_params:,}\n")


# loss, optimizer, scheduler 
pos_weight = torch.tensor([n_taxiway / n_runway]).to(device)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predicted = (outputs >= 0.0).float()
        correct   += (predicted == labels).sum().item()
        total     += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predicted = (outputs >= 0.0).float()
            correct   += (predicted == labels).sum().item()
            total     += labels.size(0)

    return running_loss / total, correct / total


# training stuffs
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss':   [], 'val_acc':   [],
}

best_val_acc        = 0.0
patience_counter    = 0
EARLY_STOP_PATIENCE = 5

print("Starting training...\n")

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)

    scheduler.step(val_loss)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f"Epoch [{epoch+1:02d}/{EPOCHS}]  "
          f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  |  "
          f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_path = os.path.join(MODELS_DIR, "assistnet_v2_1_best.pth")
        torch.save(model.state_dict(), best_path)
        print(f"  --> New best model saved (val_acc: {best_val_acc:.4f})")
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= EARLY_STOP_PATIENCE:
        print(f"\nEarly stopping triggered after {epoch+1} epochs.")
        break

# save model 
final_path = os.path.join(MODELS_DIR, "assistnet_v2_1_final.pth")
torch.save({
    'model_state_dict':     model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'history':              history,
    'class_to_idx':         train_dataset.class_to_idx,
}, final_path)

print(f"\nFinal model saved to: {final_path}")
print(f"Best model saved to:  {best_path}")

# plotting

epochs_range = range(len(history['train_acc']))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, history['train_acc'], label='Training Accuracy')
plt.plot(epochs_range, history['val_acc'],   label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history['train_loss'], label='Training Loss')
plt.plot(epochs_range, history['val_loss'],   label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.tight_layout()
plot_path = os.path.join(PLOTS_DIR, "training_curves_v2_1.png")
plt.savefig(plot_path)
plt.show()
print(f"Training plot saved to: {plot_path}")