"""
Face Mask Image Classification - Ultra-Fast PyTorch Training & Evaluation
Uses MobileNetV2 with transfer learning for fast, high-accuracy training (<1 min).
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # 1. Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    if device.type == 'cpu':
        torch.set_num_threads(min(8, torch.get_num_threads()))
        print(f"PyTorch CPU threads: {torch.get_num_threads()}", flush=True)
    
    data_dir = "Face Mask Dataset"
    train_dir = os.path.join(data_dir, "Train")
    val_dir = os.path.join(data_dir, "Validation")
    test_dir = os.path.join(data_dir, "Test")
    
    # 2. Fast transforms (Optimized for CPU throughput)
    image_size = (160, 160)
    train_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Datasets & DataLoaders
    batch_size = 128
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    class_names = train_dataset.classes
    print(f"Classes: {class_names} -> {train_dataset.class_to_idx}", flush=True)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}", flush=True)
    
    # 4. Model Definition (Transfer Learning with MobileNetV2)
    print("Loading pre-trained MobileNetV2...", flush=True)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # Freeze feature extractor for ultra-fast training
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Replace classification head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, len(class_names))
    )
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.002)
    
    # 5. Training Loop
    epochs = 2
    print(f"\n--- Starting Ultra-Fast Training for {epochs} Epochs ---", flush=True)
    start_time = time.time()
    
    best_val_acc = 0.0
    best_model_weights = None
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)
            
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}", flush=True)
                
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()
                val_total += inputs.size(0)
                
        val_loss = val_loss / val_total
        val_acc = val_corrects / val_total
        epoch_duration = time.time() - epoch_start
        
        print(f"==> Epoch {epoch} completed in {epoch_duration:.1f}s | "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%", flush=True)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = model.state_dict().copy()
            
    total_duration = time.time() - start_time
    print(f"\nTraining completed in {total_duration:.1f}s! Best Validation Accuracy: {best_val_acc*100:.2f}%", flush=True)
    
    # Load best model weights and save
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    
    model_save_path = "face_mask_mobilenetv2.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'class_to_idx': train_dataset.class_to_idx
    }, model_save_path)
    print(f"Model saved successfully to '{model_save_path}'", flush=True)
    
    # 6. Evaluation on Test Set
    print("\n--- Evaluating Model on Test Dataset ---", flush=True)
    model.eval()
    test_loss = 0.0
    test_corrects = 0
    test_total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data).item()
            test_total += inputs.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    test_acc = test_corrects / test_total
    print(f"Test Loss: {test_loss/test_total:.4f} | Test Accuracy: {test_acc*100:.2f}% ({test_corrects}/{test_total})", flush=True)
    
    print("\nClassification Report:", flush=True)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4), flush=True)
    
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm, flush=True)

if __name__ == "__main__":
    main()
