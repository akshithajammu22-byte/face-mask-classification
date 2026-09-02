"""
Face Mask Classifier - Standalone Inference Script
Classify any external image (JPG, PNG, JPEG, WebP) as 'WithMask' or 'WithoutMask'.
Usage:
    python predict.py --image "path/to/image.jpg"
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

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

def predict_image(image_input, model, class_names, device):
    """
    image_input: file path (str) or PIL Image
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        img = image_input.convert('RGB')
    else:
        raise ValueError("Unsupported image input type. Provide a file path or PIL Image.")
        
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    tensor = transform(img).unsqueeze(0).to(device)
    
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
    print(f" Image File   : {args.image}")
    status = "[MASK DETECTED]" if result['prediction'] == "WithMask" else "[NO MASK]"
    print(f" Prediction   : {status} -> {result['prediction']}")
    print(f" Confidence   : {result['confidence']*100:.2f}%")
    print("-" * 45)
    print(" Class Probabilities:")
    for cls_name, prob in result['probabilities'].items():
        bar = "#" * int(prob * 25)
        print(f"  - {cls_name:<12}: {prob*100:6.2f}% |{bar}")
    print("="*45 + "\n")

if __name__ == "__main__":
    main()
