import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


model = tf.keras.models.load_model('Brain_Tumor_Model.h5')

st.title("Brain Tumor Detection")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    img = image.load_img(uploaded_image, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    predictions = model.predict(img_array)
    if predictions[0][0] > 0.5:
        st.write("Predicted: No")
    else:
        st.write("Predicted: Yes")