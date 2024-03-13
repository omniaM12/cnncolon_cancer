import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from PIL import Image


model = joblib.load('model')
import streamlit as st
from transformers import pipeline
from PIL import Image

pipeline = Pipeline(task="image-classification", model=model)

st.title("tumor or not")

file_name = st.file_uploader("Upload a hot dog candidate image")

if file_name is not None:
    col1, col2 = st.columns(2)

    image = Image.open(file_name)
    col1.image(image, use_column_width=True)
    predictions = Pipeline(image)

    col2.header("Probabilities")
    for p in predictions:
        col2.subheader(f"{ p['label'] }: { round(p['score'] * 100, 1)}%")
