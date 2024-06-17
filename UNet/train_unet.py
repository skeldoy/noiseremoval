import os
import cv2
import numpy as np

# Paths to the datasets
clean_data_path = '../data/clean'
overlay_data_path = '../data/overlays'
output_path = '../data/cleanedup'

# Create output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

clean_images = load_images_from_folder(clean_data_path)
overlay_images = load_images_from_folder(overlay_data_path)

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    # Encoding path
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    # Decoding path
    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = concatenate([up6, conv4], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = concatenate([up7, conv3], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = concatenate([up8, conv2], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(3, 1, activation='sigmoid')(conv9)

    model = tf.keras.Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = unet_model()

def preprocess_images(images):
    processed_images = []
    for img in images:
        img = cv2.resize(img, (256, 256))
        img = img / 255.0  # Normalize the images
        processed_images.append(img)
    return np.array(processed_images)

clean_images = preprocess_images(clean_images)
overlay_images = preprocess_images(overlay_images)

model.fit(overlay_images, clean_images, epochs=50, batch_size=16, validation_split=0.1)

def save_images(images, folder):
    for i, img in enumerate(images):
        img = (img * 255).astype(np.uint8)  # Denormalize the images
        cv2.imwrite(os.path.join(folder, f'cleaned_{i}.png'), img)

# Perform inference
predicted_clean_images = model.predict(overlay_images)
save_images(predicted_clean_images, output_path)

model.save('overlay_removal_model.h5')

