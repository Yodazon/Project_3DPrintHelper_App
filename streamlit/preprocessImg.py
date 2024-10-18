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

