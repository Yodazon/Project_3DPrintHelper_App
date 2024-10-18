### Code to develop a streamlit app
### Using the model I made in CNN for 3D prints
import streamlit as st
import preprocessImg as pImg
import torch
import torchvision.models as models
import pyTorchModel as py
import os
import time

st.title("3D Printing Issue Solver")

#Function for model prediction
def predictImage(processedImage):

    class_names = {0: 'good', 1: 'spaghetti', 2: 'stringing', 3: 'underextrusion'}


    # Get the parent directory of the current directory (streamlit)
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the model weights in the CNNBuilding folder
    model_weights_path = os.path.join(parent_dir, "..", "CNNBuilding", "models_in_folder", "CNNModelV0_3.pth")


    model = py.pyTorchModel()
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))

    model.eval()
    output = model(processedImage)
    
    predicted_index = output.argmax(1).item()
    predicted_class_name = class_names[predicted_index]
    
    return predicted_class_name

#Function to store images and their predictions
#@st.cache_data
#def fetch_and_cache_image(uploadedImage,processedImage, prediction):
#Function for Sidebar




st.write("Hi there! Welcome to the 3D Print App, where we try to solve your 3D Printing issues")
st.write("")
st.write("The solving part is still under development. At the moment the program can determine what your issue is")

uploaded_image = st.file_uploader("Upload Image Here")


if uploaded_image is not None:
    st.write("Image you have uploaded is below:")
    st.image(uploaded_image, width=500)
    with st.spinner("Please Wait..."):
        time.sleep(2)
        #Calling to pre process image
        processedImage = pImg.preProcess(uploaded_image)

        #Calling the model
        prediction = predictImage(processedImage)
        st.write(f"The predicted problem is {prediction} ")
else:
    
    st.error('No file Uploaded')

#Adding dataframe for cached images?
st.dataframe()





##CODE FOR API REQUEST TO FLASK IN RENDER
import requests
st.write("The following button is an API call to render")
result = st.button("Click for Prediction")
if result:
    st.write("running")
    resp = requests.post("https://project-3dprintingapp.onrender.com", uploaded_image)
    #resp = requests.post("http://127.0.0.1:5000", uploaded_image)
    st.write("The predicted problem is")
    st.write(f"{resp}")