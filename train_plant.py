import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import timm
import json

def run_eda(dataset, prefix="plant"):


    print(f"[{prefix.upper()}] Running Exploratory Data Analysis...")
    os.makedirs("graphs", exist_ok=True)
    
    counts = Counter(dataset.targets)
    classes = dataset.classes
    plt.figure(figsize=(12, 6))
    
    # minimal class names to fit nicely
    short_classes = [c[:15]+".." if len(c)>15 else c for c in classes]
    sns.barplot(x=[short_classes[k] for k in counts.keys()], y=list(counts.values()))
    
    plt.xticks(rotation=90)
    plt.title(f"{prefix.capitalize()} Class Distribution")
    plt.tight_layout()

    plt.savefig(f"graphs/{prefix}_class_distribution.png")
    plt.close()

class FocalLoss(nn.Module):


    # Focal loss selected to address class imbalance
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        log_pt = -self.ce(inputs, targets)
        pt = torch.exp(log_pt)
        loss = -self.alpha * ((1 - pt) ** self.gamma) * log_pt
        return loss.mean()

def train_pipeline():
    base_dir = "Plant&Disease/New Plant Diseases Dataset(Augmented)"
    train_dir = os.path.join(base_dir, "train")
    if not os.path.exists(train_dir): 
        print("Plant dataset train dir not found!")
        return
    
    val_dir = os.path.join(base_dir, "valid")
    
    # Data Augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    val_data = datasets.ImageFolder(val_dir, transform=val_transform)
    
    run_eda(train_data, prefix="plant")

    print(f"[PLANT] Loading model architecture...")


    model = timm.create_model("deit_tiny_patch16_224", pretrained=True, num_classes=len(train_data.classes))
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)


    epochs = 1
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4, steps_per_epoch=250, epochs=epochs)

    
    print(f"[PLANT] Starting training (250 batches)...")

    model.train()
    running_loss = 0.0
    for batch_idx, (imgs, lbls) in enumerate(train_loader):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, lbls)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}/250 Loss: {loss.item():.4f}")
        if batch_idx >= 249: break




    print("[PLANT] Evaluating...")
    model.eval()
    all_preds, all_lbls = [], []
    with torch.no_grad():
        for batch_idx, (imgs, lbls) in enumerate(val_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_lbls.extend(lbls.cpu().numpy())
    
    acc = accuracy_score(all_lbls, all_preds)

    prec = precision_score(all_lbls, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_lbls, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_lbls, all_preds, average='weighted', zero_division=0)




    
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }
    
    
    os.makedirs("graphs", exist_ok=True)

    with open("graphs/plant_metrics.json", "w") as f:
        json.dump(metrics, f)
    
    # Confusion Matrix
    cm = confusion_matrix(all_lbls, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title("Plant Disease Validation Matrix")
    plt.savefig("graphs/plant_confusion_matrix.png")

    plt.close()




    # Save classes
    with open("plant_classes.txt", "w") as f:

        f.write("\n".join(train_data.classes))
    
    torch.save(model.state_dict(), "plant_model.pth")
    print("[PLANT] Saved plant_model.pth!")

if __name__ == "__main__":
    train_pipeline()

