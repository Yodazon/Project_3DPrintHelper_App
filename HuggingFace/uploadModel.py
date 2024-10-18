import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
import pyTorchModel_HF as py
import os

 # Get the parent directory of the current directory (HuggingFace)
parent_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the model weights in the CNNBuilding folder
model_weights_path = os.path.join(parent_dir, "..", "CNNBuilding", "models_in_folder", "CNNModelV0_3.pth")


model = py.pyTorchModel()
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
model.push_to_hub("model")
