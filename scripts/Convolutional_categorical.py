from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import ResNet50

import numpy as np
import os

input_folder = './output_cell_images'
save_directory = './models_trained/Convolutional_categorical'
model_path = os.path.join(save_directory, 'efficientnet_model.h5')
checkpoint_filepath = os.path.join(save_directory, 'best_model12.h5')

predicted_classes = []
index_to_label = {0: 'ASC-H', 1: 'ASC-US', 2: 'HSIL', 3: 'LSIL', 4: 'Negative for intraepithelial lesion', 5: 'SCC'}

img_size = (100, 100)

# Função para preparar uma imagem para classificação
def prepare_image(file_path, img_size):
    img = image.load_img(file_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Normalizando os pixels

    return img_array

# Função para fazer a previsão
def classify_image(file_path, model, img_size):
    img_ready = prepare_image(file_path, img_size)
    predictions = model.predict(img_ready)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = index_to_label[predicted_class[0]]
    print(predicted_label)
    return predicted_label

def classify_convolutional():
    global predicted_classes
    predicted_classes = []
    # Carregar modelo se existir, caso contrário, criar um novo
    if os.path.exists(checkpoint_filepath):
        print('Executando Convolucional Categórico...')
        model = load_model(checkpoint_filepath)
        diretorio_imagens = './image_nucleus'
        # Percorrer todas as imagens no diretório
        for nome_arquivo in os.listdir(diretorio_imagens):
            # Montar o caminho completo da imagem
            img_path = os.path.join(diretorio_imagens, nome_arquivo)
            # Classificar a imagem
            predicted_label = classify_image(img_path, model, img_size)
            predicted_classes.append(predicted_label)      
    else:
        
        print("Modelo não encontrado. Criando um novo modelo.")
        labels = os.listdir(input_folder)

        datagen = ImageDataGenerator(
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2,
            rescale=1./255  
        )

        batch_size = 32
        epochs = 15

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

        y_train = train_generator.classes

        print(y_train)
        

        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )

        base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(100, 100, 3))
        
        # for camada in base_model.layers:
        #     camada.trainable = False

        model = models.Sequential()
        model.add(base_model)
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(len(labels), activation='softmax'))

        model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
   
        # Cálculo dos pesos das classes
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_generator.classes),
            y=train_generator.classes
        )

        # Convertendo os pesos para um dicionário para passar durante o treinamento
        class_weights_dict = dict(enumerate(class_weights))

        # Treina o modelo
        epochs = 150
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[model_checkpoint]
        )

        model.fit(train_generator, epochs=epochs, callbacks=[model_checkpoint], validation_data=val_generator)

        model.save(model_path)      

        print('\t FIM Convolucional Categórico')

    return predicted_classes

# classify_convolutional()
