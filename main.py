import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from sklearn.metrics import confusion_matrix, accuracy_score
import wandb
import numpy as np
import os

# --- Configurations ---
WANDB_PROJECT = "mldlops-minor"
WANDB_ENTITY = "priyansh-saxena" # Matches your screenshot setup
HF_DATASET_ID = "Chiranjeev007/STL-10_Subset"
HF_MODEL_REPO = "b22ee075/stl10-resnet18-minor" # Change if needed
BATCH_SIZE = 64
EPOCHS = 5 # Adjust based on your time constraints, ResNet18 converges quickly
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

# ==========================================
# STEP 2: Custom Dataset & Transformations
# ==========================================
class STL10CustomDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image'].convert('RGB')
        label = item['label']
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    # Initialize WandB (Ensure WANDB_API_KEY is in your env vars)
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name="resnet18-stl10-run")

    # Image transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ==========================================
    # STEP 1: Load Data from HuggingFace
    # ==========================================
    print("Loading dataset...")
    dataset = load_dataset(HF_DATASET_ID)
    
    train_data = STL10CustomDataset(dataset['train'], transform=train_transform)
    val_data = STL10CustomDataset(dataset['validation'], transform=val_test_transform)
    test_data = STL10CustomDataset(dataset['test'], transform=val_test_transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # ==========================================
    # STEP 3: Use Pretrained ResNet-18
    # ==========================================
    print("Setting up model...")
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    # Modify final layer for 10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ==========================================
    # STEP 4: Training Loop & WandB Plots
    # ==========================================
    best_val_acc = 0.0
    best_model_path = "best_resnet18.pth"

    for epoch in range(EPOCHS):
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Log Train/Val metrics to WandB
        wandb.log({
            "Train Loss": train_loss, "Train Accuracy": train_acc,
            "Validation Loss": val_loss, "Validation Accuracy": val_acc,
            "Epoch": epoch + 1
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    # ==========================================
    # STEP 5: Push Best Model to HuggingFace
    # ==========================================
    print("Pushing best model to Hugging Face...")
    api = HfApi()
    api.create_repo(repo_id=HF_MODEL_REPO, exist_ok=True)
    api.upload_file(
        path_or_fileobj=best_model_path,
        path_in_repo="best_resnet18.pth",
        repo_id=HF_MODEL_REPO
    )

    # ==========================================
    # STEP 6: Load Model from HuggingFace
    # ==========================================
    print("Downloading model from Hugging Face for Evaluation...")
    downloaded_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="best_resnet18.pth")
    
    eval_model = models.resnet18()
    eval_model.fc = nn.Linear(eval_model.fc.in_features, 10)
    eval_model.load_state_dict(torch.load(downloaded_path))
    eval_model = eval_model.to(DEVICE)
    eval_model.eval()

    # ==========================================
    # Test Evaluation Prep
    # ==========================================
    all_preds, all_labels, all_images = [], [], []
    correct_samples, incorrect_samples = [], []

    # Un-normalize function for visualizing images in WandB
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = eval_model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Collect samples for Step 9
            for i in range(inputs.size(0)):
                img_tensor = inputs[i].cpu()
                pred_label = preds[i].item()
                true_label = labels[i].item()
                
                # Un-normalize and clamp for visualization
                img_vis = inv_normalize(img_tensor).clamp(0, 1).permute(1, 2, 0).numpy()

                if pred_label == true_label and len(correct_samples) < 10:
                    correct_samples.append((img_vis, pred_label, true_label))
                elif pred_label != true_label and len(incorrect_samples) < 10:
                    incorrect_samples.append((img_vis, pred_label, true_label))

    test_accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # ==========================================
    # STEP 7 & 8: Confusion Matrix & Class-wise Bar Plot on WandB
    # ==========================================
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            preds=all_preds, y_true=all_labels, class_names=CLASS_NAMES
        )
    })

    class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    bar_data = [[CLASS_NAMES[i], acc] for i, acc in enumerate(class_accuracies)]
    table = wandb.Table(data=bar_data, columns=["Class", "Accuracy"])
    wandb.log({
        "class_wise_accuracy": wandb.plot.bar(table, "Class", "Accuracy", title="Class-wise Accuracy")
    })

    # ==========================================
    # STEP 9: Show 20 Test Samples on WandB
    # ==========================================
    wandb_images = []
    for img, pred, true in correct_samples + incorrect_samples:
        caption = f"Pred: {CLASS_NAMES[pred]} | Actual: {CLASS_NAMES[true]}"
        wandb_images.append(wandb.Image(img, caption=caption))
    
    wandb.log({"Test Samples (10 Correct, 10 Incorrect)": wandb_images})

    # ==========================================
    # STEP 10: Report on Exam Sheet (Standard Output)
    # ==========================================
    print("\n" + "="*50)
    print("STEP 10: EXAM SHEET REPORTING")
    print("="*50)
    print(f"a. Test Accuracy: {test_accuracy * 100:.2f}%")
    print("b. Class-wise Accuracy for each class:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"   Class {i} ({class_name}): {class_accuracies[i] * 100:.2f}%")
    print("="*50)

    wandb.finish()

if __name__ == "__main__":
    main()