import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Flower Classification using AI", page_icon="🌸")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("flower_classifier_model.keras")


@st.cache_data
def load_class_names():
    with open("class_names.json") as f:
        return json.load(f)


model = load_model()
class_names = load_class_names()

st.title("🌸 Flower Classification using AI")
st.write(
    "Upload a flower image and the CNN model will predict whether it is a "
    "daisy, dandelion, rose, sunflower or tulip."
)

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    # Preprocess (the model rescales pixels internally)
    resized = image.resize((180, 180))
    img_array = tf.expand_dims(np.array(resized), 0)

    predictions = model.predict(img_array)[0]
    top = int(np.argmax(predictions))

    st.success("Prediction completed!")
    st.write("### Predicted Class:", class_names[top])
    st.write("### Confidence:", f"{100 * predictions[top]:.2f}%")

    st.subheader("All class probabilities")
    for i in np.argsort(predictions)[::-1]:
        st.write(f"{class_names[i]}: {100 * predictions[i]:.2f}%")
        st.progress(float(predictions[i]))
