import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img
from tensorflow.keras.models import load_model

# Parameters
input_shape = (576, 1024, 3)
img_height, img_width, _ = input_shape

# Load the trained autoencoder model
autoencoder = load_model('denoising_autoencoder.h5')

# Function to denoise an image using the trained autoencoder
def denoise_image(image_path, save_path, model):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    denoised_img_array = model.predict(img_array)
    denoised_img = array_to_img(denoised_img_array[0])
    
    denoised_img.save(save_path)

# Paths to the input and output directories
input_dir = "../data/inference"
output_dir = "../data/autoencoders"

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Denoise all images in the input directory
for img_filename in os.listdir(input_dir):
    input_img_path = os.path.join(input_dir, img_filename)
    output_img_path = os.path.join(output_dir, img_filename)
    
    # Denoise and save the image
    denoise_image(input_img_path, output_img_path, autoencoder)


