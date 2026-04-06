import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from collections import Counter
import timm
import numpy as np
import json

def run_eda(dataset, prefix="cassava"):


    print(f"[{prefix.upper()}] Exploratory Data Analysis..")
    os.makedirs("graphs", exist_ok=True)
    
    counts = Counter(dataset.targets)
    classes = dataset.classes
    plt.figure(figsize=(8, 6))
    
    sns.barplot(x=classes, y=list(counts.values()), palette="viridis")
    
    plt.xticks(rotation=45)
    plt.title("Cassava Class Distribution")
    plt.tight_layout()

    plt.savefig(f"graphs/{prefix}_class_distribution.png")
    plt.close()

def plot_roc_binary(preds_prob, all_lbls, healthy_idx, num_classes):

    # Convert labels for Healthy (1) vs Disease (0)
    binary_lbls = [1 if l == healthy_idx else 0 for l in all_lbls]
    # Predict probabilities of healthy
    prob_healthy = preds_prob[:, healthy_idx]
    
    fpr, tpr, _ = roc_curve(binary_lbls, prob_healthy)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Disease vs Healthy)')
    plt.legend(loc="lower right")
    plt.savefig("graphs/cassava_binary_roc.png")
    plt.close()

def train_pipeline():
    base_dir = "CassavaDisease"
    train_dir = os.path.join(base_dir, "train")
    if not os.path.exists(train_dir): 
        print("Cassava dataset not found in", train_dir)
        return
    
    # Data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    run_eda(train_data, prefix="cassava")

    print(f"[CASSAVA] Loading model architecture...")


    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=len(train_data.classes))
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    # Simplified train/val split since Cassava train is a single folder
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_data, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5) 
 

    epochs = 1
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4, steps_per_epoch=250, epochs=epochs)
    
    print(f"[CASSAVA] Starting training (250 batches)...")
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




    print("[CASSAVA] Evaluating model...")

    model.eval()
    all_preds_prob = []
    all_lbls = []
    all_preds = []
    with torch.no_grad():
        for batch_idx, (imgs, lbls) in enumerate(val_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds_prob.append(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_lbls.extend(lbls.cpu().numpy())
            
    all_preds_prob = np.vstack(all_preds_prob)
    
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
    with open("graphs/cassava_metrics.json", "w") as f:
        json.dump(metrics, f)
    
    print(f"[CASSAVA] Metrics: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

    healthy_idx = train_data.class_to_idx.get('healthy', 0)
    plot_roc_binary(all_preds_prob, all_lbls, healthy_idx, num_classes=len(train_data.classes))

    cm = confusion_matrix(all_lbls, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Oranges", xticklabels=train_data.classes, yticklabels=train_data.classes)
    plt.title("Cassava Validation Matrix")
    plt.savefig("graphs/cassava_confusion_matrix.png")
    plt.close()

    # Save classes
    with open("cassava_classes.txt", "w") as f:
        f.write("\n".join(train_data.classes))

    torch.save(model.state_dict(), "cassava_model.pth")

    print("[CASSAVA] Saved cassava_model.pth!")

if __name__ == "__main__":
    train_pipeline()

