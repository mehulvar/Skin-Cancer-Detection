import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Google Drive Paths
base_path = '/Users/mehulvarshney/My Drive/Grade 11/AOS Research/DATA/FINAL COMBINED DATASET ROTATED'
output_path = '/Users/mehulvarshney/My Drive/Grade 11/AOS Research/DATA/HAM 10000 GAN Images/GAN_generated/GANvascCombined'
os.makedirs(output_path, exist_ok=True)

IMG_SIZE = 64
BATCH_SIZE = 64
LATENT_DIM = 100
EPOCHS = 5000
SAVE_INTERVAL = 500

# Data Loader
def load_images(base_path):
    images = []
    print(f"Checking directory: {base_path}")
    if not os.path.exists(base_path):
        print("Error: Base path does not exist!")
        return np.array(images)

    for filename in os.listdir(base_path):
        print(f"Found file: {filename}")
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                img = load_img(os.path.join(base_path, filename), target_size=(IMG_SIZE, IMG_SIZE))
                img = img_to_array(img) / 127.5 - 1.0  # Normalize to [-1, 1]
                images.append(img)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    print(f"Total images loaded: {len(images)}")
    return np.array(images)

# Load Data
data = load_images(base_path)
if data.size == 0:
    print("No images found in the specified directory. Exiting...")
    exit()
print(f"Loaded {data.shape[0]} images.")

# Build Generator
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(8*8*256, use_bias=False, input_shape=(LATENT_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((8, 8, 256)),
        layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', use_bias=False, activation='tanh'),
    ])
    return model

# Build Discriminator
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, kernel_size=5, strides=2, padding='same', input_shape=[IMG_SIZE, IMG_SIZE, 3]),
        layers.LeakyReLU(),
        layers.Conv2D(128, kernel_size=5, strides=2, padding='same'),
        layers.LeakyReLU(),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')  # Added activation for binary classification
    ])
    return model

# Compile GAN
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

z = layers.Input(shape=(LATENT_DIM,))
img = generator(z)
discriminator.trainable = False
validity = discriminator(img)
combined = tf.keras.Model(z, validity)
combined.compile(optimizer='adam', loss='binary_crossentropy')

# Training Loop
def train(data):
    valid = np.ones((BATCH_SIZE, 1))
    fake = np.zeros((BATCH_SIZE, 1))

    for epoch in range(EPOCHS):
        idx = np.random.randint(0, data.shape[0], BATCH_SIZE)
        real_imgs = data[idx]

        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        g_loss = combined.train_on_batch(noise, valid)

        if epoch % SAVE_INTERVAL == 0:
            save_images(epoch)
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}%] [G loss: {g_loss}]")

# Save Generated Images
def save_images(epoch):
    noise = np.random.normal(0, 1, (5, LATENT_DIM))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to [0, 1]
    for i, img in enumerate(gen_imgs):
        plt.imsave(f"{output_path}/generated_{epoch}_{i}.png", img)

# Train the GAN
train(data)
print("Training Complete!")
