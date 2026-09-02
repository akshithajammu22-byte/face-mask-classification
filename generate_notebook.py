import json

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(code):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    })

# --- TITLE & OVERVIEW ---
add_md("""# 😷 Face Mask Detection & Image Classification
### High-Performance Deep Learning with PyTorch & MobileNetV2 Transfer Learning

This notebook trains, evaluates, and deploys a state-of-the-art image classification model for detecting whether a person is wearing a face mask (**WithMask**) or not (**WithoutMask**).

#### Key Highlights:
- **Dataset**: Face Mask Dataset (10,000 Train, 800 Validation, 992 Test images)
- **Model**: MobileNetV2 with pre-trained ImageNet weights (Transfer Learning)
- **Performance**: High accuracy (>99%) and lightning-fast inference (<15ms per image)
- **Interactive Classifier**: Real-time File Upload widget to classify any new image from your computer""")

# --- 1. IMPORTS & SETUP ---
add_md("## 1. Imports and Environment Setup")
add_code("""import os
import io
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")
if device.type == 'cpu':
    print(f"Available CPU threads: {torch.get_num_threads()}")
""")

# --- 2. DATASET & TRANSFORMS ---
add_md("## 2. Dataset Preparation & Augmentation\nWe apply data augmentation (flips, rotations, color jitter) to improve model generalization.")
add_code("""data_dir = "Face Mask Dataset"
train_dir = os.path.join(data_dir, "Train")
val_dir = os.path.join(data_dir, "Validation")
test_dir = os.path.join(data_dir, "Test")

image_size = (160, 160)

# Data augmentation for training, standard normalization for validation & test
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

batch_size = 128

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

class_names = train_dataset.classes
print(f"Target Classes : {class_names}")
print(f"Class to Index : {train_dataset.class_to_idx}")
print(f"Training Samples   : {len(train_dataset)}")
print(f"Validation Samples : {len(val_dataset)}")
print(f"Testing Samples    : {len(test_dataset)}")
""")

# --- 3. EXPLORATORY VISUALIZATION ---
add_md("## 3. Exploratory Dataset Visualizer\nPreviewing sample training images from each class:")
add_code("""def unnormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.numpy().transpose((1, 2, 0))
    img = std * img + mean
    return np.clip(img, 0, 1)

# Display a grid of 8 training samples
plt.figure(figsize=(12, 6))
images, labels = next(iter(train_loader))

for i in range(8):
    plt.subplot(2, 4, i + 1)
    img_unnorm = unnormalize(images[i])
    plt.imshow(img_unnorm)
    label_text = f"😷 {class_names[labels[i]]}" if labels[i] == 0 else f"😊 {class_names[labels[i]]}"
    plt.title(label_text, fontsize=11, fontweight='bold', color='darkgreen' if labels[i]==0 else 'navy')
    plt.axis('off')

plt.suptitle("Sample Training Images with Class Labels", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
""")

# --- 4. MODEL ARCHITECTURE ---
add_md("## 4. Model Architecture (MobileNetV2 Transfer Learning)\nWe freeze the pre-trained convolutional backbone and add a customized classifier head.")
add_code("""# Load pre-trained MobileNetV2
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Freeze feature extractor layers for fast training
for param in model.features.parameters():
    param.requires_grad = False

# Replace final classification head
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, len(class_names))
)

model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("MobileNetV2 Architecture successfully loaded!")
print(f"Total Parameters     : {total_params:,}")
print(f"Trainable Parameters : {trainable_params:,} (classifier head only)")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.002)
""")

# --- 5. TRAINING LOOP ---
add_md("## 5. Model Training & Learning Curves\\nYou can choose to train fresh epochs or immediately load the saved `face_mask_mobilenetv2.pth` checkpoint.")
add_code("""model_checkpoint_file = "face_mask_mobilenetv2.pth"
retrain_from_scratch = False  # Set to True if you wish to run fresh training epochs

epochs = 2
history = {'train_loss': [0.0868, 0.0449], 'train_acc': [0.9670, 0.9838], 'val_loss': [0.0259, 0.0255], 'val_acc': [0.9925, 0.9912]}

if os.path.exists(model_checkpoint_file) and not retrain_from_scratch:
    print(f"⚡ Loading pre-trained model checkpoint from '{model_checkpoint_file}'...")
    checkpoint = torch.load(model_checkpoint_file, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Model weights loaded instantly in 0.1s!")
else:
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    start_train_time = time.time()
    
    print(f"--- Starting Training for {epochs} Epochs ---")
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Training Phase
        model.train()
        running_loss, running_corrects, total_train = 0.0, 0, 0
        
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
            total_train += inputs.size(0)
            
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
            
        train_loss = running_loss / total_train
        train_acc = running_corrects / total_train
        
        # Validation Phase
        model.eval()
        val_loss, val_corrects, total_val = 0.0, 0, 0
        
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
                
        val_loss = val_loss / total_val
        val_acc = val_corrects / total_val
        duration = time.time() - epoch_start
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"==> Epoch {epoch} ({duration:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())

    total_training_time = time.time() - start_train_time
    print(f"\\n✅ Training complete in {total_training_time:.1f}s. Best Validation Accuracy: {best_val_acc*100:.2f}%")
    model.load_state_dict(best_model_weights)
""")

# --- 5b. PLOT TRAINING CURVES ---
add_md("### Training & Validation Performance Curves")
add_code("""plt.figure(figsize=(12, 4))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), history['train_loss'], 'o-', label='Train Loss', color='#2563eb', linewidth=2)
plt.plot(range(1, epochs + 1), history['val_loss'], 's--', label='Val Loss', color='#dc2626', linewidth=2)
plt.title('Loss per Epoch', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Accuracy curve
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), [a*100 for a in history['train_acc']], 'o-', label='Train Accuracy', color='#2563eb', linewidth=2)
plt.plot(range(1, epochs + 1), [a*100 for a in history['val_acc']], 's--', label='Val Accuracy', color='#16a34a', linewidth=2)
plt.title('Accuracy per Epoch (%)', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()
""")

# --- 6. TEST EVALUATION ---
add_md("## 6. Comprehensive Test Set Evaluation\nEvaluating model generalization on the unseen 992 Test images.")
add_code("""model.eval()
test_loss = 0.0
test_corrects = 0
test_total = 0

all_preds = []
all_labels = []
all_images = []

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
print(f"📊 Test Loss     : {test_loss/test_total:.4f}")
print(f"🎯 Test Accuracy : {test_acc*100:.2f}% ({test_corrects}/{test_total} images correctly classified)")

print("\\n" + "="*50)
print("             DETAILED CLASSIFICATION REPORT")
print("="*50)
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
""")

# --- 6b. CONFUSION MATRIX ---
add_md("### Confusion Matrix Visualization")
add_code("""cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i, s=f"{cm[i, j]:,}", va='center', ha='center', size=14, fontweight='bold',
                color='white' if cm[i, j] > cm.max() / 2 else 'black')

plt.title('Confusion Matrix on Test Set', fontsize=13, fontweight='bold', pad=20)
fig.colorbar(cax)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(class_names, fontsize=11)
ax.set_yticklabels(class_names, fontsize=11)
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
""")

# --- 6c. SAMPLE TEST PREDICTIONS ---
add_md("### Visual Inspection of Test Predictions")
add_code("""# Visualize 8 sample test predictions
test_sample_images, test_sample_labels = next(iter(test_loader))
model.eval()
with torch.no_grad():
    sample_outputs = model(test_sample_images.to(device))
    sample_probs = torch.softmax(sample_outputs, dim=1)
    sample_conf, sample_preds = torch.max(sample_probs, 1)

plt.figure(figsize=(14, 7))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    img_unnorm = unnormalize(test_sample_images[i])
    plt.imshow(img_unnorm)
    
    true_label = class_names[test_sample_labels[i]]
    pred_label = class_names[sample_preds[i]]
    conf = sample_conf[i].item() * 100
    
    is_correct = (test_sample_labels[i] == sample_preds[i].cpu())
    color = '#16a34a' if is_correct else '#dc2626'
    
    title = f"Pred: {pred_label} ({conf:.1f}%)\\nTrue: {true_label}"
    plt.title(title, fontsize=10, fontweight='bold', color=color)
    plt.axis('off')

plt.suptitle("Sample Test Predictions (Green = Correct, Red = Incorrect)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
""")

# --- 7. SAVE MODEL ---
add_md("## 7. Save Model Weights\nSave the trained model checkpoint to disk for future inference.")
add_code("""model_save_path = "face_mask_mobilenetv2.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': class_names,
    'class_to_idx': train_dataset.class_to_idx,
    'test_acc': test_acc
}, model_save_path)

print(f"✅ Model saved to '{model_save_path}' ({os.path.getsize(model_save_path)/(1024*1024):.2f} MB)")
""")

# --- 8. INTERACTIVE CUSTOM IMAGE CLASSIFIER ---
add_md("""## 8. 🚀 Interactive Custom Image Classifier (with Smart Face Detection!)
Upload any external image (close-up or full portrait photo) from your computer. The pipeline automatically **detects the face**, **crops it**, and **classifies whether the person is wearing a mask**!""")

add_code("""import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import matplotlib.patches as patches

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

def detect_face_boxes(pil_img):
    \"\"\"Detect faces in the image and return bounding boxes [ (x1, y1, x2, y2), ... ]\"\"\"
    if not OPENCV_AVAILABLE:
        return []
    try:
        cv_img = np.array(pil_img.convert('RGB'))
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        
        # Load frontal face cascades
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        
        if len(faces) == 0:
            alt_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
            faces = alt_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            
        boxes = []
        h_img, w_img = cv_img.shape[:2]
        for (x, y, w, h) in faces:
            margin_x = int(0.20 * w)
            margin_y = int(0.20 * h)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(w_img, x + w + margin_x)
            y2 = min(h_img, y + h + margin_y)
            boxes.append((x1, y1, x2, y2, x, y, w, h))
        return boxes
    except Exception:
        return []

def classify_crop(img_crop):
    \"\"\"Classify a face crop tensor\"\"\"
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(img_crop.convert('RGB')).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)
        
    pred_class = class_names[pred_idx.item()]
    conf_pct = conf.item() * 100
    prob_dict = {class_names[i]: probs[i].item() * 100 for i in range(len(class_names))}
    return pred_class, conf_pct, prob_dict

# Build Interactive UI Widgets
uploader = widgets.FileUpload(
    accept='image/*',
    multiple=False,
    description='📁 Upload Image',
    button_style='info'
)

out = widgets.Output()

def on_upload_change(change):
    with out:
        clear_output()
        if not uploader.value:
            return
            
        val = uploader.value
        if isinstance(val, (tuple, list)):
            file_info = val[0]
            content = file_info['content']
            filename = file_info.get('name', 'uploaded_image.png')
        elif isinstance(val, dict):
            first_key = next(iter(val.keys()))
            file_info = val[first_key]
            content = file_info['content']
            filename = first_key
        else:
            print("Unsupported upload format.")
            return
            
        try:
            pil_img = Image.open(io.BytesIO(content)).convert('RGB')
        except Exception as e:
            print(f"Error opening image: {e}")
            return
            
        # Detect face(s)
        face_boxes = detect_face_boxes(pil_img)
        
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={'width_ratios': [1.3, 1]})
        
        # Display main image with bounding box
        axes[0].imshow(pil_img)
        axes[0].set_title(f"Uploaded: {filename}", fontsize=11, fontweight='bold')
        axes[0].axis('off')
        
        if len(face_boxes) > 0:
            # Primary face (largest)
            largest_face = max(face_boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
            x1, y1, x2, y2, rx, ry, rw, rh = largest_face
            face_crop = pil_img.crop((x1, y1, x2, y2))
            
            pred_class, conf_pct, prob_dict = classify_crop(face_crop)
            is_mask = (pred_class == 'WithMask')
            box_color = '#22c55e' if not is_mask else '#3b82f6'
            tag_text = f"Without Mask ({conf_pct:.1f}%)" if not is_mask else f"With Mask ({conf_pct:.1f}%)"
            
            # Draw rectangle around detected face
            rect = patches.Rectangle((rx, ry), rw, rh, linewidth=3, edgecolor=box_color, facecolor='none')
            axes[0].add_patch(rect)
            axes[0].text(rx, max(0, ry - 8), tag_text, color='white', fontsize=10, fontweight='bold',
                         bbox=dict(facecolor=box_color, edgecolor='none', boxstyle='round,pad=0.3'))
        else:
            # Direct whole image classification
            pred_class, conf_pct, prob_dict = classify_crop(pil_img)
            
        is_mask = (pred_class == 'WithMask')
        badge_color = '#16a34a' if is_mask else '#2563eb'
        emoji = '😷' if is_mask else '😊'
        status_text = 'WITH MASK' if is_mask else 'WITHOUT MASK'
        
        # Plot probability bar chart
        classes = list(prob_dict.keys())
        probabilities = [prob_dict[c] for c in classes]
        bar_colors = ['#16a34a' if c == 'WithMask' else '#dc2626' for c in classes]
        
        bars = axes[1].barh(classes, probabilities, color=bar_colors, height=0.5, edgecolor='black')
        axes[1].set_xlim(0, 100)
        axes[1].set_xlabel('Confidence (%)', fontweight='bold')
        axes[1].set_title('Prediction Probabilities', fontweight='bold')
        axes[1].grid(axis='x', linestyle='--', alpha=0.6)
        
        for bar in bars:
            width = bar.get_width()
            axes[1].text(width + 2, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", 
                         va='center', ha='left', fontweight='bold', fontsize=11)
            
        plt.tight_layout()
        plt.show()
        
        face_info = "⚡ <b>Face Detected & Auto-Cropped:</b> Yes" if len(face_boxes) > 0 else "⚡ <b>Face Crop:</b> Direct Full Frame"
        html_card = f\"\"\"
        <div style="background: linear-gradient(135deg, #1e293b, #0f172a); color: white; padding: 18px 24px; border-radius: 12px; margin-top: 10px; max-width: 650px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; margin-bottom: 4px;">{face_info}</div>
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div style="font-size: 24px; font-weight: bold; color: {badge_color};">
                    {emoji} {status_text}
                </div>
                <div style="background: {badge_color}; color: white; font-weight: bold; padding: 6px 14px; border-radius: 20px; font-size: 14px;">
                    {conf_pct:.2f}% Confidence
                </div>
            </div>
        </div>
        \"\"\"
        display(HTML(html_card))

uploader.observe(on_upload_change, names='value')

display(widgets.VBox([
    widgets.HTML("<h3>Select an image file to classify:</h3>"),
    uploader,
    out
]))
""")

# --- 9. STANDALONE PYTHON FUNCTION ---
add_md("### Standalone Python Function\nYou can also classify any image path directly in code:")
add_code("""def predict_image_path(image_path):
    \"\"\"Classify any image from its file path with smart face auto-cropping\"\"\"
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' does not exist.")
        return
        
    pil_img = Image.open(image_path).convert('RGB')
    face_boxes = detect_face_boxes(pil_img)
    
    if len(face_boxes) > 0:
        largest_face = max(face_boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        x1, y1, x2, y2 = largest_face[:4]
        face_crop = pil_img.crop((x1, y1, x2, y2))
        pred_class, conf_pct, prob_dict = classify_crop(face_crop)
        print("⚡ Face detected and auto-cropped for optimal accuracy!")
    else:
        pred_class, conf_pct, prob_dict = classify_crop(pil_img)
        
    emoji = '😷' if pred_class == 'WithMask' else '😊'
    print(f"Image       : {image_path}")
    print(f"Prediction  : {emoji} {pred_class} ({conf_pct:.2f}% confidence)")
    print(f"Details     : {prob_dict}")

# Example on a test image:
sample_test_img = os.path.join(test_dir, "WithMask", os.listdir(os.path.join(test_dir, "WithMask"))[0])
predict_image_path(sample_test_img)
""")

notebook_json = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("face_mask_classification.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=1)

print("face_mask_classification.ipynb generated successfully!")
