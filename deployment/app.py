import gradio as gr
import torch
from torchvision import transforms
from torchvision.models import resnet50
import pytorch_lightning as pl
from PIL import Image
import json

class ImageNetModule(pl.LightningModule):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.model = resnet50()
        self.model.fc = torch.nn.Linear(2048, num_classes)
    
    def forward(self, x):
        return self.model(x)

# Load model
def load_model():
    model = ImageNetModule.load_from_checkpoint("model.ckpt")
    model.eval()
    return model

# Preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Load ImageNet classes with fallback
def load_imagenet_labels():
    try:
        with open('imagenet_classes.json') as f:
            labels = json.load(f)
        # Ensure we have all 1000 classes
        for i in range(1000):
            if str(i) not in labels:
                labels[str(i)] = f"Class {i}"
        return labels
    except Exception as e:
        print(f"Error loading labels: {e}")
        return {str(i): f"Class {i}" for i in range(1000)}

labels = load_imagenet_labels()

def predict(image):
    if image is None:
        return {f"Error: No image provided": 1.0}
    
    try:
        # Preprocess image
        img = transform(Image.fromarray(image))
        img = img.unsqueeze(0)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(img)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get top 5 predictions
        top5_prob, top5_catid = torch.topk(probs, 5)
        
        results = []
        for i in range(5):
            class_id = str(top5_catid[i].item())
            class_name = labels.get(class_id, f"Class {class_id}")
            results.append((class_name, float(top5_prob[i])))
        
        return {label: conf for label, conf in results}
    except Exception as e:
        return {f"Error: {str(e)}": 1.0}

# Create Gradio interface
try:
    model = load_model()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=5),
    examples=[
        ["examples/cat.jpg"],
        ["examples/dog.jpg"],
        ["examples/bird.jpg"]
    ],
    title="ImageNet Classification with ResNet50",
    description="Upload an image to classify it into one of 1000 ImageNet categories.",
    cache_examples=False  # Disable example caching to prevent errors
)

if __name__ == "__main__":
    interface.launch() 