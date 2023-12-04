import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import cv2
import os
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import nucleus_detection
import plot_graphs
from sklearn.metrics import accuracy_score

training_data_directory = './cell_images'
save_directory = './models_trained/Mahalanobis_binary'

characteristics_and_predicted_classes = []
characteristics_and_classes = []
predicted_classes = []  # Lista para armazenar as classes previstas
accuracy = ""

# Calcular a distância de Mahalanobis
def calculate_mahalanobis(class_means_dict, class_covariance_dict):
    global characteristics_and_predicted_classes
    characteristics_and_predicted_classes = []
    for characteristics, _ in characteristics_and_classes:
        min_distance = float('inf')
        predicted_class = None
        for classe in class_means_dict:
            cov_inv = np.linalg.inv(class_covariance_dict[classe])
            mean_vector = class_means_dict[classe]
            mahalanobis_distance = mahalanobis(characteristics, mean_vector, cov_inv)
            if mahalanobis_distance < min_distance:
                min_distance = mahalanobis_distance
                predicted_class = classe
        predicted_classes.append(predicted_class)

        characteristics_and_predicted_classes.append([characteristics[0], characteristics[1], characteristics[2], predicted_class]) 
    
def classify_mahalanobis_binary(image_cell_path, nucleus_info):  
    covariance_path = os.path.join(save_directory, 'class_covariance.joblib')
    means_path = os.path.join(save_directory, 'class_means.joblib')

    if os.path.exists(covariance_path) and os.path.exists(means_path):
        print('Executando Mahalanobis Binário...')
        # Carregando as matrizes de covariância e médias
        class_covariance_dict = load(covariance_path)
        class_means_dict = load(means_path)
        
        # Redefinindo os valores das variáveis a cada execução da função
        global characteristics_and_classes
        global predicted_classes 
        
        characteristics_and_classes = nucleus_info
        predicted_classes = []  

        # Calculando a partir das caracteristicas extraidas dos núcleos       
        calculate_mahalanobis(class_means_dict, class_covariance_dict)
    else:
        # Se não existe
        print('Os arquivos de matrizes de covariância e/ou médias não existem no diretório.')
        print('Calculando as métricas...')
        conteudo = os.listdir(training_data_directory)
        
        # Percorrendo todas as imagens no diretório de treino
        for image in conteudo:
            print(f'imagem: {image}')
            image_path = os.path.join(training_data_directory, image)
            features = nucleus_detection.get_characteristics(image_path)
            for characteristic in features:
                excentricidade = characteristic[1]
                area = characteristic[2]
                compacidade = characteristic[3]
                classe = characteristic[4]
                if(classe != 'Negative for intraepithelial lesion'):
                    classe = 'Positive for intraepithelial lesion'
                # Area minima para considerar na contagem das métricas
                if area > 100:                        
                    characteristics_and_classes.append(([area, excentricidade, compacidade], classe)) 
            
        # Calcular a média e a matriz de covariância para cada classe
        class_covariance_dict = {}
        class_means_dict = {} 
            
        for characteristics, classe in characteristics_and_classes:
            if classe not in class_covariance_dict:
                class_covariance_dict[classe] = []
                class_means_dict[classe] = []
        
            class_covariance_dict[classe].append(characteristics)
            class_means_dict[classe].append(characteristics)
        
        # Calculando as médias e matrizes de covariância
        for classe in class_covariance_dict:
            observations = np.array(class_covariance_dict[classe])
            class_covariance_dict[classe] = np.cov(observations, rowvar=False)
            class_means_dict[classe] = np.mean(observations, axis=0)
        
        # Carregar as matrizes de covariância e as médias salvas
        covariance_path = os.path.join(save_directory, 'class_covariance.joblib')
        means_path = os.path.join(save_directory, 'class_means.joblib')

        # Salvando as matrizes de covariância e as médias
        dump(class_covariance_dict, covariance_path)
        dump(class_means_dict, means_path)
        
        print('Treinamento finalizado!!')

    true_classes = [classe for _, classe in characteristics_and_classes]
    data = dict();
    data['characteristics_and_classes'] = characteristics_and_classes
    data['characteristics_and_predicted_classes'] = characteristics_and_predicted_classes
    data['predicted_classes'] = predicted_classes
    data['true_classes'] = true_classes
    data['accuracy'] = accuracy_score(true_classes, predicted_classes)
    
    print('\t FIM Mahalanobis Binário')

    return data

    

