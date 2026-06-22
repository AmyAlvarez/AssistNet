import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, classification_report

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

ROOT       = r"C:\Users\Your_Path\AssistNet\AssistNet_v2"
TEST_DIR   = os.path.join(ROOT, "data", "test")
MODELS_DIR = os.path.join(ROOT, "models")
PLOTS_DIR  = os.path.join(ROOT, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

# device check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

eval_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# test dataset & load
test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transforms)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

class_to_idx = test_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}
class_names  = [idx_to_class[i] for i in range(len(idx_to_class))]

print(f"\nClass mapping: {class_to_idx}")
print(f"Test set: {len(test_dataset)} images\n")

# model
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
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),

        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model_path = os.path.join(MODELS_DIR, "assistnet_v2_best.pth")

model = AssistNetV2().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(f"Loaded model from: {model_path}\n")

all_preds  = []
all_labels = []
all_probs  = []

# use BCEWithLogitsLoss to match training takes raw logits as input very cool
criterion = nn.BCEWithLogitsLoss()
running_loss, correct, total = 0.0, 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(images)             # raw logits
        loss    = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        # convert logit to probability for display/reporting
        prob       = torch.sigmoid(outputs).item()
        # threshold at 0.0 logit = 0.5 probability
        predicted  = int(outputs.item() >= 0.0)
        true_label = int(labels.item())

        all_probs.append(prob)
        all_preds.append(predicted)
        all_labels.append(true_label)

        correct += (predicted == true_label)
        total   += 1

test_loss = running_loss / total
test_acc  = correct / total

print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}  ({correct}/{total} correct)\n")

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

def plot_confusion_matrix(cm, classes,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


cm = confusion_matrix(all_labels, all_preds)
plot_confusion_matrix(cm, class_names)

cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.show()
print(f"Confusion matrix saved to: {cm_path}")

# sample pedications
def unnormalize(tensor):
    """Reverse ImageNet normalization for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


plt.figure(figsize=(12, 12))

sample_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

with torch.no_grad():
    taxiway_idx = test_dataset.class_to_idx['taxiway']
    taxiway_shown = 0

    for i, (images, labels) in enumerate(sample_loader):
        if taxiway_shown >= 9:
            break
        # skip runway images
        if labels.item() != taxiway_idx:
            continue

        images = images.to(device)
        output = model(images)
        prob   = torch.sigmoid(output).item()
        pred   = int(output.item() >= 0.0)
        true   = labels.item()

        img = unnormalize(images.squeeze(0).cpu())
        img = img.permute(1, 2, 0).numpy()

        plt.subplot(3, 3, taxiway_shown + 1)
        plt.imshow(img)
        plt.axis('off')

        pred_label = idx_to_class[pred]
        true_label = idx_to_class[true]
        color = 'green' if pred == true else 'red'
        plt.title(f"Pred: {pred_label} ({prob:.2f})\nTrue: {true_label}",
                  color=color, fontsize=9)

        taxiway_shown += 1

plt.suptitle('Sample Taxiway Test Predictions  (green = correct, red = wrong)',
             fontsize=12)
plt.tight_layout()

grid_path = os.path.join(PLOTS_DIR, "sample_predictions.png")
plt.savefig(grid_path)
plt.show()
print(f"Sample predictions saved to: {grid_path}")

# misclassified images

print("\nFinding misclassified images...")

misclassified = []

with torch.no_grad():
    for i, (images, labels) in enumerate(sample_loader):
        images = images.to(device)
        output = model(images)
        prob   = torch.sigmoid(output).item()
        pred   = int(output.item() >= 0.0)
        true   = labels.item()

        if pred != true:
            misclassified.append({
                'image': images.squeeze(0).cpu(),
                'prob': prob,
                'pred': pred,
                'true': true,
                'path': test_dataset.imgs[i][0]  # original file path
            })

print(f"Found {len(misclassified)} misclassified images\n")
for m in misclassified:
    print(f"  File: {os.path.basename(m['path'])}")
    print(f"  True: {idx_to_class[m['true']]}  Pred: {idx_to_class[m['pred']]}  Confidence: {m['prob']:.4f}\n")

# plot dem 
n_misc = len(misclassified)
if n_misc > 0:
    cols = min(n_misc, 4)
    rows = (n_misc + cols - 1) // cols
    plt.figure(figsize=(4 * cols, 4 * rows))

    for i, m in enumerate(misclassified):
        img = unnormalize(m['image'])
        img = img.permute(1, 2, 0).numpy()

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Pred: {idx_to_class[m['pred']]} ({m['prob']:.2f})\n"
                  f"True: {idx_to_class[m['true']]}",
                  color='red', fontsize=9)

    plt.suptitle(f'All Misclassified Images ({n_misc} total)', fontsize=12)
    plt.tight_layout()

    misc_path = os.path.join(PLOTS_DIR, "misclassified.png")
    plt.savefig(misc_path)
    plt.show()
    print(f"Misclassified images saved to: {misc_path}")
else:
    print("No misclassifications — perfect test set!")