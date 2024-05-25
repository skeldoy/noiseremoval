import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img
import numpy as np

# Suppresses INFO and WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set memory growth to avoid GPU memory issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load the trained generator model
generator = load_model('generator_model.h5')

# Directory containing noisy images to be denoised
input_dir = "../data/training"
output_dir = "../data/cyclegan"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

def preprocess_image(image_path):
    """Load and preprocess an image for the generator model."""
    image = load_img(image_path, target_size=(576, 1024))
    image = img_to_array(image)
    image = (image / 127.5) - 1.0  # Normalize to [-1, 1]
    return image

def postprocess_image(image):
    """Postprocess the image from the generator output."""
    image = (image + 1.0) * 127.5  # Denormalize to [0, 255]
    image = tf.clip_by_value(image, 0, 255)
    return image

def denoise_image(image_path, output_path):
    """Denoise an image and save the result."""
    # Preprocess the image
    image = preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Perform inference
    denoised_image = generator.predict(image)
    denoised_image = denoised_image[0]  # Remove batch dimension

    # Postprocess the image
    denoised_image = postprocess_image(denoised_image)

    # Convert to PIL image and save
    denoised_image = array_to_img(denoised_image)
    denoised_image.save(output_path)

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        denoise_image(input_path, output_path)
        print(f'Denoised image saved to: {output_path}')

