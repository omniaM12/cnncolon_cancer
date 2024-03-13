import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
 

model = releases/tag/v.0.1.0-model




st.title("tumor or not")

file_name = st.file_uploader("Upload a histopathology image")

if file_name is not None:
    col1, col2 = st.columns(2)

    image = Image.open(file_name)
    col1.image(image, use_column_width=True)
    predictions = model.predict(image)

    col2.header("Probabilities")
    for p in predictions:
        col2.subheader(f"{ p['label'] }: { round(p['score'] * 100, 1)}%")
