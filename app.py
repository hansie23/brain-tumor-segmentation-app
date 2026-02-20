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
    image = Image.open(image).convert('RGB')
    image = image.resize((112,112))
    new_image_array = np.array(image) / 255
    expanded_image_array = np.expand_dims(new_image_array, axis=0)
    return expanded_image_array

@st.cache_resource
def get_model(token):
    # Download the model from Hugging Face Hub
    model_weights_path = hf_hub_download(
        repo_id="hansie23/brain-tumor-segmentation-model",
        filename="optimized_unet_checkpoint.keras",
        token=token
    )
    model = load_model_and_weights(model_weights_path)
    return model

def main():
    st.title('Brain Tumor Segmentation in MRI scans')

    # Get the token from environment variables (Streamlit Cloud or local environment)
    hugging_face_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if not hugging_face_token:
        st.error("Hugging Face token not found. Please ensure 'HUGGING_FACE_HUB_TOKEN' is set in your environment or Streamlit Secrets.")
        return

    uploaded_image = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png", "tif"])

    if st.button('Predict'):
        if uploaded_image is not None:
            with st.spinner('Downloading and loading model...'):
                model = get_model(hugging_face_token)
            
            preprocessed_image = preprocess_image(uploaded_image)
            
            with st.spinner('Analyzing image...'):
                prediction = model.predict(preprocessed_image)

            col1, col2 = st.columns(2)
            col1.image(uploaded_image, caption='Uploaded Image', width="stretch")
            # Display only the first image in the batch and remove batch dimension
            col2.image(prediction[0], caption='Predicted Image', width="stretch")
        else:
            st.warning('Please upload an image.')

if __name__ == '__main__':
    main()
