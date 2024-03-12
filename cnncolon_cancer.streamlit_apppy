import streamlit as st
import pandas as pd
import numpy as np
import os
from tensorflow.keras.activations import softmax
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from tensorflow.keras import preprocessing
import tensorflow as tf
from tensorflow.keras.models import load_model
import h5py

model = joblib.load('model')
st.header("Colon cancer predictor")
def main():
    file_uploaded= st.file_uploader("choose the file", type =["jpg","png","jpeg"])
    if file_uploaded is not None:
        image= Image.open(file_uploaded)
        figure= plt.figure()
        plt.imshow(image)
        plt.axis("off")
        result= predict.Class(image)
        st.write(result)
        st.pyplot(figure)
def predict_class(image):
    classifier_model= tf.keras.models.load_model("model")
    shape=((150,150,3))
    model=tf.keras.Sequential(hub[hub.kerasLayer(classifier_model,input_shape=shape)])
    test_image= image.resize((150,150))
    test_image= preprocessing.image.img_to_array(test_image)
    test_image= test_image/255.0
    test_image= np.expand_dim(test_image,axis=0)
    class_names=["no_tumor","tumor"]
    predictions=model.predict(test_image)
    scores=tf.nn.softmax(predictions[0])
    scores= scores.numpy()
    image_class=class_names[np.argmax(scores)]
    result= "the image uploaded is: {}".format(image_class)
    return result
if _ _ name _ _== "_ _main_ _":
   main()
 
