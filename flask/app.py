import io
import os
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
import pyTorchModel as py
from flask import Flask, jsonify, request

app = Flask(__name__)

# Get the parent directory of the current directory (streamlit)
parent_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the model weights in the CNNBuilding folder
model_weights_path = os.path.join(parent_dir, "..", "CNNBuilding", "models_in_folder", "CNNModelV0_2.pth")

model = py.pyTorchModel()
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
model.eval()


def preProcess(image):
    # Load and preprocess the image
    image = Image.open(image).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize(227),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    input_image = transform(image).unsqueeze(0)

    return input_image

def prediction(image):
    class_names = {0: 'good', 1: 'spaghetti', 2: 'stringing', 3: 'underextrusion'}
    tensor = preProcess(image=image)
    output = model(tensor)

    predicted_index = output.argmax(1).item()
    predicted_class_name = class_names[predicted_index]
    
    return predicted_class_name





@app.route('/predict', methods=['POST'])
def predict():
    if request.method =='POST':
        file = request.files['file']
        image_raw = file.read()
        class_result = prediction(image = image_raw)
    
        return class_result

if __name__ == '__main__':
    app.run()
