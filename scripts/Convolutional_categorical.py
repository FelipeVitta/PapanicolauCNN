from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras import regularizers
import numpy as np
import os

input_folder = './output_cell_images'
save_directory = './models_trained/Convolutional_categorical'
model_path = os.path.join(save_directory, 'efficientnet_model.h5')

predicted_classes = []

# Função para preparar uma imagem para classificação
def prepare_image(file_path, img_size):
    img = image.load_img(file_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Normalizando os pixels

    return img_array

# Função para fazer a previsão
def classify_image(file_path, model, img_size, labels):
    img_ready = prepare_image(file_path, img_size)
    predictions = model.predict(img_ready)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = labels[predicted_class[0]]

    return predicted_label

labels = os.listdir(input_folder)

datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

batch_size = 32
img_size = (100, 100)

train_generator = datagen.flow_from_directory(
    input_folder,
    class_mode='categorical',
    target_size=img_size,
    batch_size=batch_size,
    subset='training',
    classes=labels
)

val_generator = datagen.flow_from_directory(
    input_folder,
    class_mode='categorical',
    target_size=img_size,
    batch_size=batch_size,
    subset='validation',
    classes=labels
)

# Carregar modelo se existir, caso contrário, criar um novo
if os.path.exists(model_path):
    print("Modelo carregado com sucesso.")
    model = load_model(model_path)
    diretorio_imagens = './image_nucleus'
     # Percorrer todas as imagens no diretório
    for nome_arquivo in os.listdir(diretorio_imagens):
        # Montar o caminho completo da imagem
        img_path = os.path.join(diretorio_imagens, nome_arquivo)
        # Classificar a imagem
        predicted_label = classify_image(img_path, model, img_size, labels)
        print(f"Classe prevista para {nome_arquivo}: {predicted_label}")
        predicted_classes.append(predicted_label)
        
    #print(predicted_classes)
else:
    print("Modelo não encontrado. Criando um novo modelo.")
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(100, 100, 3))

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(len(labels), activation='softmax'))

    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    # Treina o modelo
    epochs = 7
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator
    )

    model.save(model_path)


