import ai_edge_torch
import numpy
import torch
import torchvision
import pyTorchModel as py
import os
import torchvision.transforms as transforms
from PIL import Image








##Load pyTorch model and create Edge Model
def load_model():
    # Get the parent directory of the current directory (streamlit)
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the model weights in the CNNBuilding folder
    model_weights_path = os.path.join(parent_dir, "..", "models_in_folder", "CNNModelV0_2.pth")


    model = py.pyTorchModel()
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('gpu')))


    edge_model = ai_edge_torch.convert(model.eval())

    return model, edge_model


#Determine if pytorch and edge model are the same
def check_Model_Tolerance(torch_output, edge_output):
    if (numpy.allclose(
        torch_output.detach().numpy(),
        edge_output,
        atol=1e-5,
        rtol=1e-5,
    )):
        print("Inference result with Pytorch and TfLite was within tolerance")
    else:
        print("Something wrong with Pytorch --> TfLite")


#Inference for single Image PyTorch
def inference_single_image(model, image_path):
    image_size = 227
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    input_image = transform(image).unsqueeze(0)  # Add batch dimension

    # Set the model to evaluation mode and make predictions
    model.eval()
    with torch.no_grad():
        input_image = input_image.to("cuda")
        output = model(input_image)
    


    return output, input_image



# pyTorchModel, edgeModel = load_model()




# image_path = "C:\\Coding\\Github\\Project_3DPrintingApp\\CNNBuilding\\test_image\\spag.jpg"
# output, image_input = inference_single_image(pyTorchModel, image_path)
