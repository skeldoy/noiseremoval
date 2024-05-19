import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

# Define the autoencoder architecture
input_img = Input(shape=(256, 256, 3))  # Example input shape

# Encoder
x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Load data (example data, replace with your own dataset)
# Assuming you have a dataset of clean images X_train and X_test
# For demonstration, let's create random images
X_train = np.random.rand(1000, 256, 256, 3)
X_test = np.random.rand(100, 256, 256, 3)

# Function to add noise to images
def add_noise(images):
    noise_factor = 0.5
    noisy_images = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    noisy_images = np.clip(noisy_images, 0., 1.)
    return noisy_images

# Create noisy images
X_train_noisy = add_noise(X_train)
X_test_noisy = add_noise(X_test)

# Train the autoencoder
autoencoder.fit(X_train_noisy, X_train,
                epochs=10,
                batch_size=32,
                shuffle=True,
                validation_split=0.2)

# Denoise test images
decoded_imgs = autoencoder.predict(X_test_noisy)

# Display original, noisy, and denoised images for comparison (using matplotlib for visualization)
import matplotlib.pyplot as plt

n = 10  # Number of images to display
plt.figure(figsize=(20, 6))
for i in range(n):
    # Display original images
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i])
    plt.title("Original")
    plt.axis("off")

    # Display noisy images
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(X_test_noisy[i])
    plt.title("Noisy")
    plt.axis("off")

    # Display denoised images
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs[i])
    plt.title("Denoised")
    plt.axis("off")

plt.show()

