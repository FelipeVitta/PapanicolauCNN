from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import models, layers, optimizers
import numpy as np
import os

input_folder = './output_cell_images'
save_directory = './models_trained/Convolutional_categorical'

class_mapping = {
    'ASC-H': 'Positive for intraepithelial lesion',
    'ASC-US': 'Positive for intraepithelial lesion',
    'HSIL': 'Positive for intraepithelial lesion',
    'LSIL': 'Positive for intraepithelial lesion',
    'SCC': 'Positive for intraepithelial lesion',
    'Negative for intraepithelial lesion': 'Negative for intraepithelial lesion',
}

dataset = []
labels = []

for folder_name in os.listdir(input_folder):
    if folder_name in class_mapping:
        class_label = class_mapping[folder_name]
        folder_path = os.path.join(input_folder, folder_name)

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = load_img(image_path, target_size=(100, 100))
            image = img_to_array(image)

            dataset.append(image)
            labels.append(class_label)


dataset = np.array(dataset)
dataset = dataset / 255.0

le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

X_train, X_val, y_train, y_val = train_test_split(dataset, labels_categorical, test_size=0.2, random_state=42)

train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
val_generator = val_datagen.flow(X_val, y_val, batch_size=32)

# Modelo EfficientNet
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(100, 100, 3))

model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(np.unique(labels_encoded)), activation='softmax'))

model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Treinando o modelo
epochs = 1
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator
)

model.save('efficientnet_model.h5')