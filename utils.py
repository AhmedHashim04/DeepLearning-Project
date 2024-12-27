import os
from PIL import Image
import torch

def allowed_file(filename, allowed_extensions={'png', 'jpg', 'jpeg'}):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def load_model(model_path, num_classes, device):
    from torchvision import models
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, data_transform, classes, device):
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = data_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted_class = torch.max(outputs, 1)
        return classes[predicted_class.item()]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error"
