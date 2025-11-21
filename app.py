import streamlit as st
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import os

from unet_model import unet_model

def load_model_and_weights(model_weights_path):
    model = unet_model()
    model.load_weights(model_weights_path)
    return model

def preprocess_image(image):
    image = Image.open(image)
    image = image.resize((112,112))
    new_image_array = np.array(image) / 255
    expanded_image_array = np.expand_dims(new_image_array, axis=0)
    return expanded_image_array

def main():
    st.title('Brain Tumor Segmentation in MRI scans')

    # Read Hugging Face token from the mounted secret file
    hugging_face_token_path = "/run/secrets/hugging_face_token"
    if os.path.exists(hugging_face_token_path):
        with open(hugging_face_token_path, "r") as f:
            hugging_face_token = f.read().strip()
    else:
        hugging_face_token = None

    if not hugging_face_token:
        st.error("Hugging Face token not found. Please ensure it's provided as a Docker build secret.")
        return

    uploaded_image = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png", "tif"])

    if st.button('Predict'):
        if uploaded_image is not None:
            # Download the model from Hugging Face Hub
            model_weights_path = hf_hub_download(
                repo_id="hansie23/brain-tumor-segmentation-model",
                filename="optimized_unet_checkpoint.keras",
                use_auth_token=hugging_face_token
            )
            
            model = load_model_and_weights(model_weights_path)
            preprocessed_image = preprocess_image(uploaded_image)
            prediction = model.predict(preprocessed_image)

            col1, col2 = st.columns(2)
            col1.image(uploaded_image, caption='Uploaded Image', use_container_width=True)
            col2.image(prediction, caption='Predicted Image', use_container_width=True)
        else:
            st.warning('Please upload an image.')

if __name__ == '__main__':
    main()
