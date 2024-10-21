from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from torchvision import transforms
from PIL import Image
import os
from io import BytesIO
import pyTorchModel as py
import uvicorn

# Get the parent directory of the current directory (streamlit)
parent_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the model weights in the CNNBuilding folder
model_weights_path = os.path.join(parent_dir, "..", "CNNBuilding", "models_in_folder", "CNNModelV0_3.pth")

model = py.pyTorchModel()
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
model.eval()


class_names = {0: 'good', 1: 'spaghetti', 2: 'stringing', 3: 'underextrusion'}

app = FastAPI()

class Prediction(BaseModel):
    class_name: str
    confidence: float


def preProcess(image):
    # Open the image from raw bytes
    image = Image.open(BytesIO(image)).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize(227),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    input_image = transform(image).unsqueeze(0)
    return input_image



@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # Read and preprocess the image
    image = await file.read()
    input_tensor = preProcess(image)

    # Disable gradient calculation for faster inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()

    predicted_class = probabilities.argmax()
    class_name = class_names[predicted_class]
    confidence = probabilities[predicted_class]


    return {"class_name": class_name, "confidence": confidence}

# Run the server with: uvicorn main:app --reload


if __name__ == "__main__":
    uvicorn.run("my_app:app", host="0.0.0.0", port=10000, reload=True)