import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING logs

import tensorflow as tf

# Set memory growth to avoid GPU memory issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, Activation
from tensorflow.keras.models import Model

# Define the generator model
def build_generator():
    inputs = Input(shape=(576, 1024, 3))
    
    # Encoder
    x = Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, (4, 4), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(256, (4, 4), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Decoder
    x = Conv2DTranspose(128, (4, 4), strides=2, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(64, (4, 4), strides=2, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(3, (4, 4), strides=2, padding='same')(x)
    outputs = Activation('tanh')(x)
    
    model = Model(inputs, outputs)
    return model

# Define the discriminator model
def build_discriminator():
    inputs = Input(shape=(576, 1024, 3))
    
    x = Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, (4, 4), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(256, (4, 4), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(512, (4, 4), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    outputs = Conv2D(1, (4, 4), padding='same')(x)
    
    model = Model(inputs, outputs)
    return model

# Initialize the models
G = build_generator()
D = build_discriminator()

# Define the loss and optimizers
def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Example training step
@tf.function
def train_step(noisy_images):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = G(noisy_images, training=True)
        
        real_output = D(noisy_images, training=True)
        fake_output = D(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        
    gradients_of_generator = gen_tape.gradient(gen_loss, G.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, D.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, G.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, D.trainable_variables))

image_data_dir = "../data"
batch_size = 8  # Use a smaller batch size to avoid memory issues
img_height = 576
img_width = 1024

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    image_data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Normalize pixel values to [-1, 1]
def normalize(img):
    img = (img / 127.5) - 1
    return img

# Normalize images
dataset = dataset.map(lambda x, _: normalize(x))
dataset = dataset.shuffle(buffer_size=1000)

epochs = 20
# Training loop
for epoch in range(epochs):
    for image_batch in dataset:  # Use only the image tensors
        train_step(image_batch)
    print(f'Epoch {epoch+1}/{epochs} completed')

# Save the generator model
G.save('generator_model.h5')

