from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from collections import Counter
import os

input_folder = './output_cell_images'
save_directory = './models_trained/Convolutional_binary'
model_path = os.path.join(save_directory, 'efficientnet_model.h5')
checkpoint_path = os.path.join(save_directory, "model_checkpoint2.h5")

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
predicted_classes = []
img_size = (100, 100)
index_to_label = {0: 'Negative for intraepithelial lesion', 1: 'Positive for intraepithelial lesion'}

# Função para preparar uma imagem para classificação
def prepare_image(file_path, img_size):
    img = load_img(file_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Normalizando os pixels

    return img_array

# Função para fazer a previsão
def classify_image(file_path, model, img_size):
    img_ready = prepare_image(file_path, img_size)
    predictions = model.predict(img_ready)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = index_to_label[predicted_class[0]]
    
    return predicted_label

def classify_convolutional_binary():
    global predicted_classes
    predicted_classes = []

    if os.path.exists(checkpoint_path):
        print('Executando Convolucional Binário...')
        model = load_model(checkpoint_path)
        diretorio_imagens = './image_nucleus'
        # Percorrer todas as imagens no diretório
        for nome_arquivo in os.listdir(diretorio_imagens):
            # Montar o caminho completo da imagem
            img_path = os.path.join(diretorio_imagens, nome_arquivo)
            # Classificar a imagem
            predicted_label = classify_image(img_path, model, img_size)
            # print(f"Classe prevista para {nome_arquivo}: {predicted_label}")
            predicted_classes.append(predicted_label)
            
    else: 
        for folder_name in os.listdir(input_folder):
            if folder_name in class_mapping:
                class_label = class_mapping[folder_name]
                folder_path = os.path.join(input_folder, folder_name)

                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    image = load_img(image_path, target_size=img_size)
                    image = img_to_array(image)
                    image /= 255.0  # Normalizando os pixels

                    dataset.append(image)
                    labels.append(class_label)
                    
        # Contagem das imagens para cada classe
        # class_counts = Counter(labels)
        # for class_label, count in class_counts.items():
        #     print(f"Classe '{class_label}': {count} imagens")

        dataset = np.array(dataset)
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        labels_categorical = to_categorical(labels_encoded)
        labels_unique = np.unique(labels_encoded)

        # Criar um dicionário para mapear de volta os índices para rótulos de classe
        #index_to_label = {i: label for i, label in enumerate(label_encoder.classes_)}

        X_train, X_val, y_train, y_val = train_test_split(dataset, labels_categorical, test_size=0.2, random_state=42)

        train_datagen = ImageDataGenerator(
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

        val_datagen = ImageDataGenerator()

        train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
        val_generator = val_datagen.flow(X_val, y_val, batch_size=32)

        # Calcula os pesos das classes de forma que classes com menos amostras tenham um peso maior
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=labels_unique,
            y=labels_encoded
        )

        class_weights_dict = dict(enumerate(class_weights))
        print("Pesos das Classes:", class_weights_dict)

    
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min',
            save_weights_only=False
        )

        
        print('Modelo não encontrado. Criando um novo modelo...')
        base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(100, 100, 3))

        model = models.Sequential()
        model.add(base_model)
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(len(np.unique(labels_encoded)), activation='softmax'))

        model.compile(optimizer=optimizers.Adam(learning_rate=0.001), 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])

        # Treinando o modelo com a ponderação de classes
        epochs = 50
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            class_weight=class_weights_dict,  # Aplicando a ponderação aqui
            callbacks=[checkpoint]  # Inclui o checkpoint aqui
        )

        model.save(model_path)

    print('\t FIM Convolucional Binário')

    return predicted_classes
