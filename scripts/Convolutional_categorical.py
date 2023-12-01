from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import models, layers, optimizers
import os

# Define your input folder
input_folder = './output_cell_images'
save_directory = './models_trained/Convolutional_categorical'

# Get subfolder names as labels
labels = os.listdir(input_folder)
print(labels)

# # Manually split the labels to ensure each label is present in both sets
# train_labels, val_labels = train_test_split(labels, test_size=0.2, random_state=42)

# print(train_labels)
# print(val_labels)

# Define image data generator with augmentation
datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split data into training and validation
)

# Create generators for training and validation sets
batch_size = 32
img_size = (100, 100)

train_generator = datagen.flow_from_directory(
    input_folder,
    class_mode='categorical',  # Change to categorical for multi-class
    target_size=img_size,
    batch_size=batch_size,
    subset='training',
    classes=labels  # Specify the subfolders to include in training
)

val_generator = datagen.flow_from_directory(
    input_folder,
    class_mode='categorical',  # Change to categorical for multi-class
    target_size=img_size,
    batch_size=batch_size,
    subset='validation',
    classes=labels  # Specify the subfolders to include in validation
)

# Create EfficientNet-based model
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(100, 100, 3))

model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(labels), activation='softmax'))  # Adjust output layer based on the number of classes

# Compile the model
model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 40
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator
)

# Save the model
model.save('efficientnet_model.h5')