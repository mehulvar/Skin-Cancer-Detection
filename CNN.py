import os
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Path to the dataset folder
data_dir = '/Users/mehulvarshney/My Drive/Grade 11/AOS Research/DATA/use this'

def is_valid_image(file_path):
    """Check if a file is a valid image."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        return False

def validate_images(directory, validate=True):
    """Validate and clean invalid images from the dataset folder."""
    if not validate:
        print("Skipping image validation.")
        return

    start_time = time.time()
    total_files = sum([len(files) for _, _, files in os.walk(directory)])
    count = 0

    print("Starting image validation...")
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)

        if os.path.isdir(folder_path):
            print(f"Entering folder: {folder}")  # Debugging statement to show folder being processed
            
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                count += 1

                if count % 100 == 0:
                    elapsed_time = time.time() - start_time
                    print(f"Checked {count}/{total_files} files. Elapsed time: {elapsed_time:.2f} seconds")

                if not is_valid_image(file_path):
                    print(f"Removing invalid file: {file_path}")
                    os.remove(file_path)

    print(f"Validation completed. Total files checked: {count}. Time taken: {time.time() - start_time:.2f} seconds.")

# Validate and clean the dataset only if required
validate_images(data_dir, validate=True)

# Validate dataset structure and determine the number of classes
class_folders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]
num_classes = len(class_folders)
print(f"Number of classes (subfolders): {num_classes}")
print(f"Class labels: {class_folders}")

# Image dimensions and other configurations
img_height, img_width = 128, 128
batch_size = 32

# Data preprocessing using ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    preprocessing_function=lambda x: tf.image.per_image_standardization(x)
)

# Train and validation datasets
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# Building a CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPool2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPool2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Adjusted to use correct num_classes
])

# Model summary
model.summary()

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model checkpointing
checkpoint_path = "/Users/mehulvarshney/Library/CloudStorage/GoogleDrive-898611@lcps.org/.My Drive/Grade 11/AOS Research/DATA/FINAL COMBINED DATASET/skin_cancer_best_model.h5"
checkpoint_callback = ModelCheckpoint(
    checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1
)

# Training the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    batch_size=batch_size,
    callbacks=[checkpoint_callback]
)

# Save the final model for future use
model_save_path = "/Users/mehulvarshney/My Drive/Grade 11/AOS Research/DATA/FINAL COMBINED DATASET/skin_cancer_model_final.h5"
model.save(model_save_path)
print(f"Final model saved to {model_save_path}")

# Get predictions on the validation data
predictions = model.predict(val_data, batch_size=batch_size)

# Get the true labels
true_labels = val_data.classes  # True labels (class indices)

# Convert predicted probabilities to class indices
predicted_labels = np.argmax(predictions, axis=1)

# Print the first 10 true and predicted labels
print("True labels:", true_labels[:10])
print("Predicted labels:", predicted_labels[:10])

# Evaluate the model's performance
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy:.4f}")

# Calculate confusion matrix
#conf_matrix = confusion_matrix(true_labels, predicted_labels)
#print("Confusion Matrix:")
#print(conf_matrix)