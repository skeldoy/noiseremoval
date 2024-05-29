import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img
import numpy as np
import os

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_dir, img_height, img_width, batch_size):
        self.image_paths = [os.path.join(image_dir, fname) for fname in 
os.listdir(image_dir)]
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_image_paths = self.image_paths[idx * self.batch_size: (idx +
1) * self.batch_size]
        images = []
        for image_path in batch_image_paths:
            img = load_img(image_path, target_size=(self.img_height, 
self.img_width))
            img = img_to_array(img) / 255.0  # Normalize to [0, 1]
            images.append(img)
        images = np.array(images)
        return images, images


# Define the autoencoder architecture
def build_autoencoder(input_shape):
    input_img = Input(shape=input_shape)

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    return autoencoder

# Load and preprocess the dataset
def load_dataset(image_dir, img_height, img_width):
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
    images = []
    for image_path in image_paths:
        img = load_img(image_path, target_size=(img_height, img_width))
        img = img_to_array(img) / 255.0  # Normalize to [0, 1]
        images.append(img)
    images = np.array(images)
    return images

# Parameters
image_dir = "../data/training"  # Path to the noisy images directory
img_height = 576
img_width = 1024
input_shape = (img_height, img_width, 3)
batch_size = 1  
epochs = 40

generator = DataGenerator(image_dir, img_height, img_width, batch_size)

# Build the autoencoder
autoencoder = build_autoencoder(input_shape)
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.MeanAbsoluteError(reduction='sum_over_batch_size',name='mean_absolute_error')) #MAE or MSE ??

# Load the dataset
#noisy_images = load_dataset(image_dir, img_height, img_width)

# Train the autoencoder
#autoencoder.fit(noisy_images, noisy_images, epochs=epochs, batch_size=batch_size, shuffle=True)

autoencoder.fit(generator, epochs=epochs, steps_per_epoch=len(generator))


# Save the autoencoder model
autoencoder.save('denoising_autoencoder.h5')

# Function to denoise an image using the trained autoencoder
def denoise_image(image_path, save_path, model):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    denoised_img_array = model.predict(img_array)
    denoised_img = array_to_img(denoised_img_array[0])
    
    denoised_img.save(save_path)

# Example usage
#denoise_image('noisy_image.jpg', 'denoised_image.jpg', autoencoder)

