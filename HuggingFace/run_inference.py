from transformers import AutoTokenizer, AutoModelForImageClassification
import torch
import os
from dotenv import load_dotenv
load_dotenv()

HF_API_KEY = os.getenv('HF_KEY')


import torchvision.transforms as transforms
from PIL import Image


image_size = 227

def preProcess(uploaded_image):
        # Load and preprocess the image
    image = Image.open(uploaded_image).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    input_image = transform(image).unsqueeze(0)

    return input_image



model_name = 'Yodazon/3DPrintFailureType'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token = HF_API_KEY)
model = AutoModelForImageClassification.from_pretrained(model_name, use_auth_token = HF_API_KEY)

image_url = "3d-print-fail.jpg"
preProcessed = preProcess(image_url)
input_file = preProcessed

inputs = tokenizer(input_file, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits

probabilities = torch.nn.functional.softmax(logits, dim=-1)

print (probabilities)