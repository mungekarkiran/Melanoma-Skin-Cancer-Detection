import streamlit as st
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image

# Function to preprocess the image (including the inverse_masked_area)
def preprocess_image(image_path):
    """
    Preprocess the image and return the inverse masked area.

    Parameters:
        image_path (str): Path to the input image.

    Returns:
        inverse_masked_image (ndarray): Image with inverse masked area.
        processing_stages_path (str): Path to saved 'processing_stages.png'.
    """
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert to 256x256 grayscale
    gray_image = cv2.cvtColor(cv2.resize(image, (256, 256)), cv2.COLOR_BGR2GRAY)

    # Apply BM3D filtering
    denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 10, 7, 21)

    # Apply binary thresholding
    _, binary_mask = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Perform morphological operations
    kernel = np.ones((5, 5), np.uint8)
    morphed_image = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Extract the masked area
    masked_area = cv2.bitwise_and(gray_image, gray_image, mask=morphed_image)

    # Extract the inverse of the masked area
    inverse_mask = cv2.bitwise_not(morphed_image)
    inverse_masked_area = cv2.bitwise_and(gray_image, gray_image, mask=inverse_mask)

    # Save all stages for visualization
    processing_stages_path = "processing_stages.png"
    stages = np.hstack((gray_image, morphed_image, masked_area, inverse_masked_area))
    cv2.imwrite(processing_stages_path, stages)

    return inverse_masked_area, processing_stages_path

# Prediction function using VGG16
def predict(image, model):
    """
    Predict using a pre-trained VGG16 model.

    Parameters:
        image (ndarray): Preprocessed image (inverse masked area).
        model (keras.Model): Pre-trained VGG16 model.

    Returns:
        str: Prediction result (e.g., 'Benign' or 'Malignant').
    """
    # Resize the image to match VGG16 input size (224x224)
    image_resized = cv2.resize(image, (224, 224))

    # Convert to 3-channel image (VGG16 expects RGB input)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)

    # Convert to array and preprocess for VGG16
    image_array = img_to_array(image_rgb)
    image_array = np.expand_dims(image_array, axis=0)
    image_preprocessed = preprocess_input(image_array)

    # Predict using the model
    predictions = model.predict(image_preprocessed)

    # Assume binary classification: class 0 = 'Benign', class 1 = 'Malignant'
    predicted_class = np.argmax(predictions, axis=1)[0]
    return "Malignant" if predicted_class == 1 else "Benign"

# Streamlit app
def main():
    st.title("Melanoma Skin Cancer Detection")
    st.write("Upload an image to detect melanoma skin cancer.")

    # Input image folder
    input_folder = "uploaded_images"
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)

    # Load pre-trained VGG16 model
    # base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    # # Add custom classification layers
    # x = base_model.output
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(256, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.5)(x)
    # predictions = tf.keras.layers.Dense(2, activation='softmax')(x)
    # vgg16_model = Model(inputs=base_model.input, outputs=predictions)
    # vgg16_model.load_weights("/content/processed_resnet50_model.h5")  # Load custom-trained weights

    vgg16_model = load_model(r'/content/processed_resnet50_model.h5')

    # File uploader
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded image
        image_path = os.path.join(input_folder, uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Image saved at {image_path}")

        # Display the uploaded image
        st.image(image_path, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        st.write("Processing the image...")
        inverse_masked_image, processing_stages_path = preprocess_image(image_path)

        # Show processing stages
        st.write("Processing stages:")
        st.image(processing_stages_path, caption="Processing Stages", use_column_width=True)

        # Perform prediction
        st.write("Predicting...")
        result = predict(inverse_masked_image, vgg16_model)

        # Display the prediction result
        st.write(f"**Prediction Result:** {result}")

# Run the app
if __name__ == "__main__":
    main()


'''
!npm install localtunnel

import urllib
print("Password/Enpoint IP for localtunnel is:",urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip("\n"))

# !streamlit run app.py --server.address=localhost &>/content/logs.txt & npx localtunnel --port 8501
!streamlit run app.py & npx localtunnel --port 8501


'''