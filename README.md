
# Car Classification

## Overview
This project is a web application that uses deep learning to classify car images. Users can upload photos of cars, and the system will identify the car model using a pre-trained ResNet50 model.

## Features
- Real-time car image upload and preview
- Instant car model prediction
- User-friendly interface
- Responsive design for all devices

## Technologies Used
- **Frontend:**
  - HTML5
  - CSS3
  - JavaScript
- **Backend:**
  - Python
  - Flask
- **Machine Learning:**
  - PyTorch
  - Pre-trained ResNet50 model

## Project Structure
```
car-classification-project/
├── app.py 
<<<<<<< HEAD
├── utils.py
=======
├── utils.py 
>>>>>>> 255a4100fa9f7ddcf96d1fda5e9ddec6f0769cb4
├── static/
│   ├── js/
│   │   └── deepcars.js           # JavaScript functions
│   ├── css/
│   │   └── style.css             # Styling
│   └── imgs/                     # Static images
├── templates/
│   └── deepCars.html             # Main HTML template
├── requirements.txt              # List of dependencies
└── models/                       # Saved model weights
├── train_model.py                # Model training script (optional)
```

## Installation
Follow the steps below to set up the project:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AhmedHashim04/DeepLearning-Project.git
   cd Car-Classification-Brand-main
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   ```
   Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Open your browser and navigate to:**
   ```
   http://localhost:5000
   ```


## Usage
1. Open the application in your web browser.
2. Click on the upload button or drag and drop a car image.
3. Wait for the model to process the image.
4. View the prediction results.

## Model Training (Optional)
If you'd like to train the model yourself:
1. Update the dataset paths in `train_model.py`.
2. Run the training script:
   ```bash
   python train_model.py
   ```
3. The trained model will be saved in the `models/` directory as `model_resnet50.pth`.

---

Enjoy using the Car Classification application!
