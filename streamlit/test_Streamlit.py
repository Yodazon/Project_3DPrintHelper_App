#This script is to determine why the main file "Streamlit_App is failing"
#August 28 2024
#This script is the same as in the "practice_Streamlit" repo 
#Titled "Main.py"

import streamlit as st
import requests
from dotenv import load_dotenv
import os
import json
load_dotenv()


st.write("hello World!")
API_key =os.getenv('API_KEY')

country = 'Canada'
api_url = 'https://api.api-ninjas.com/v1/inflation?country={}'.format(country)
response = requests.get(api_url, headers={'X-Api-Key': API_key})
if response.status_code == requests.codes.ok:
    data = json.loads(response.text)
    data[0]['yearly_rate_pct']
else:
    st.write("Error:", response.status_code, response.text)


import numpy as np
import pandas as pd

if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    chart_data


if st.checkbox('show me boobies'):
    st.write('what boobs haha!')