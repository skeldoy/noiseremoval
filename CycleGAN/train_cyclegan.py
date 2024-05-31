import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, Activation
from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator


image_data_dir = "../data"
batch_size = 6  # Use a smaller batch size to avoid memory issues
img_height = 576
img_width = 1024


datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Apply data augmentation to the dataset
dataset = datagen.flow_from_directory(
    image_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode=None
)


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

def generator_loss(fake_output, real_images, generated_images):
    gan_loss = tf.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)
    l1_loss = tf.reduce_mean(tf.abs(real_images - generated_images))  # L1 loss
    total_loss = gan_loss + (100 * l1_loss)
    return total_loss

generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

# Example training step
@tf.function
def train_step(noisy_images):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = G(noisy_images, training=True)
        
        real_output = D(noisy_images, training=True)
        fake_output = D(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output,noisy_images,generated_images)
        disc_loss = discriminator_loss(real_output, fake_output)
        
    gradients_of_generator = gen_tape.gradient(gen_loss, G.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, D.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, G.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, D.trainable_variables))

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

epochs = 60
# Training loop
for epoch in range(epochs):
    for image_batch in dataset:  # Use only the image tensors
        train_step(image_batch)
    print(f'Epoch {epoch+1}/{epochs} completed')

# Save the generator model
G.save('generator_model.h5')

