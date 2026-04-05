import os
import json
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from tensorflow.keras import layers, models

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "..", "plant_disease_model2_v1.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")

# --- 2. LOAD CLASS NAMES ---
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

# --- 3. REBUILD AND LOAD MODEL ---
# This manual rebuild avoids the 'quantization_config' error on Python 3.13
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(class_indices), activation='softmax')
])

# Load only the saved weights
model.load_weights(model_path)
print("✅ Model weights loaded successfully!")

# --- 4. PREPROCESSING FUNCTIONS ---
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# --- 5. STREAMLIT UI ---
top_col1, top_col2 = st.columns([1, 5])

with top_col1:
    # Note: Ensure this file exists in your 'app/images' folder
    if os.path.exists("images/leaf.png"):
        st.image("images/leaf.png", width=50)

with top_col2:
    st.markdown("<h1 style='margin-top: 10px;'>Plant Disease Classifier</h1>", unsafe_allow_html=True)

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        if st.button('Classify'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {prediction}')