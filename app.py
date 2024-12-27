from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from utils import allowed_file, load_model, predict_image
from torchvision import transforms
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "models/model_resnet50.pth"

classes = ['Audi', 'Lamborghini', 'Mercedes']  # Update with actual class names

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = load_model(model_path, len(classes), device)

@app.route('/')
def home():
    return render_template('deepCars.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('deepCars.html', error="No file uploaded")
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return render_template('deepCars.html', error="Invalid file type")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    prediction = predict_image(model, filepath, data_transform, classes, device)
    return render_template('deepCars.html', prediction=prediction, uploaded_image=filepath)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
