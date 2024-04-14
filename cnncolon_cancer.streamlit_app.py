import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
 

model ="model.keras"



st.title("colon tumor or normal")

file_name = st.file_uploader("Upload a histopathology image")

if file_name is not None:
    col1, col2 = st.columns(2)

    img = Image.open(file_name)
    col1.image(img, use_column_width=True)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255
    predictions = model.predict(img)[0][0]
    if pred <= 0.5:
        result = 'Predicted : normal'
    else:
        result = 'Predicted : tumor'


    col2.header("Probabilities")
    for p in predictions:
        col2.subheader(f"{ p['label'] }: { round(p['score'] * 100, 1)}%")
