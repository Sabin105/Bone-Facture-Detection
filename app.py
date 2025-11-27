import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import joblib

model = joblib.load("xgb_model.pt")

def get_label(img):
    img_g = img.convert("L")

    img_res = img_g.resize((100,100))

    img_a = np.array(img_res).flatten()

    img_df = pd.DataFrame(img_a).T

    pre = model.predict(img_df)
    return pre


st.title("Bone Fracture Detection")
st.header("A computer vision project")
file = st.file_uploader("Upload your file", type = 'png')

try: 
    if file is not None: 
        # read image 
        img = Image.open(file)
        # show Image 
        st.image(img, "The uploaded image")
        prediction = get_label(img)
        if prediction == 0:
            prediction = "Fractured"
        else:
            prediction = "Non_Fractured"
        st.write(f"The Bone is : {prediction}")

        

    else: 
        st.write("Empty File cannot be read")



except Exception as e: 
    st.write(f"{e} occured")

finally: 
    st.write("Thanks for using our service")
