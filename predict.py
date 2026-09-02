"""
Face Mask Classifier - Standalone Inference Script with Smart Face Detection
Classify any external image (close-up or full portrait) as 'WithMask' or 'WithoutMask'.
Usage:
    python predict.py --image "path/to/image.jpg"
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import numpy as np

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

def load_trained_model(weights_path="face_mask_mobilenetv2.pth", device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model checkpoint '{weights_path}' not found! Please train the model first.")
        
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    class_names = checkpoint.get('class_names', ['WithMask', 'WithoutMask'])
    
    # Rebuild architecture
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, len(class_names))
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, class_names, device

def detect_faces(pil_img):
    """Detect faces in an image and return bounding boxes [ (x1, y1, x2, y2), ... ]"""
    if not OPENCV_AVAILABLE:
        return []
        
    try:
        cv_img = np.array(pil_img.convert('RGB'))
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        
        if len(faces) == 0:
            # Try alternative cascade
            alt_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
            faces = alt_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            
        boxes = []
        h_img, w_img = cv_img.shape[:2]
        for (x, y, w, h) in faces:
            # 20% margin around face to match dataset
            margin_x = int(0.20 * w)
            margin_y = int(0.20 * h)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(w_img, x + w + margin_x)
            y2 = min(h_img, y + h + margin_y)
            boxes.append((x1, y1, x2, y2))
            
        return boxes
    except Exception:
        return []

def classify_single_crop(img, model, class_names, device):
    """Classify a single cropped face or image"""
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    tensor = transform(img.convert('RGB')).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, pred_idx = torch.max(probabilities, dim=0)
        
    pred_class = class_names[pred_idx.item()]
    prob_dict = {class_names[i]: float(probabilities[i].item()) for i in range(len(class_names))}
    
    return {
        'prediction': pred_class,
        'confidence': float(confidence.item()),
        'probabilities': prob_dict
    }

def predict_image(image_input, model, class_names, device):
    """
    Classify full image with automatic face detection and cropping.
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        img = image_input.convert('RGB')
    else:
        raise ValueError("Unsupported image input type. Provide a file path or PIL Image.")
        
    # Detect faces
    face_boxes = detect_faces(img)
    
    if len(face_boxes) > 0:
        # Evaluate primary detected face (largest)
        largest_box = max(face_boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        face_crop = img.crop(largest_box)
        result = classify_single_crop(face_crop, model, class_names, device)
        result['face_detected'] = True
        result['face_box'] = largest_box
        result['total_faces'] = len(face_boxes)
    else:
        # Fallback to direct classification (for already cropped images or full masks)
        result = classify_single_crop(img, model, class_names, device)
        result['face_detected'] = False
        result['face_box'] = None
        result['total_faces'] = 0
        
    return result

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Classify image for face mask detection")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--model", type=str, default="face_mask_mobilenetv2.pth", help="Path to trained model checkpoint")
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image '{args.image}' does not exist.")
        sys.exit(1)
        
    print(f"Loading model from {args.model}...")
    model, class_names, device = load_trained_model(args.model)
    
    result = predict_image(args.image, model, class_names, device)
    
    print("\n" + "="*45)
    print("        FACE MASK CLASSIFICATION RESULT      ")
    print("="*45)
    print(f" Image File     : {args.image}")
    print(f" Face Detected  : {'Yes (auto-cropped)' if result['face_detected'] else 'No (classified directly)'}")
    status = "[MASK DETECTED]" if result['prediction'] == "WithMask" else "[NO MASK]"
    print(f" Prediction     : {status} -> {result['prediction']}")
    print(f" Confidence     : {result['confidence']*100:.2f}%")
    print("-" * 45)
    print(" Class Probabilities:")
    for cls_name, prob in result['probabilities'].items():
        bar = "#" * int(prob * 25)
        print(f"  - {cls_name:<12}: {prob*100:6.2f}% |{bar}")
    print("="*45 + "\n")

if __name__ == "__main__":
    main()
