# 🌸 Flower Classification using CNN and Streamlit

A deep learning web app that classifies flower images into 5 classes:
**daisy, dandelion, roses, sunflowers, tulips**.

A Convolutional Neural Network (CNN) built with TensorFlow/Keras is trained on the
TensorFlow `flower_photos` dataset and served through a Streamlit interface.

## Features
- Upload a JPG/PNG image and get an instant prediction
- Confidence score and probabilities for every class
- Training notebook included (`flower_classification.ipynb`)

## Model
- 3 × (Conv2D + MaxPooling) layers, Flatten, Dense(128), softmax output
- Input size 180×180, trained for 5 epochs

## Project Structure
```
├── app.py
├── flower_classifier_model.keras
├── class_names.json
├── flower_classification.ipynb
└── requirements.txt
```

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack
Python, TensorFlow/Keras, Streamlit, NumPy, Pillow
