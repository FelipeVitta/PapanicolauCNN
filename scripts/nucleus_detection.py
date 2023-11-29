import cv2
import os
import shutil
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carregando csv
df = pd.read_csv('classifications.csv')

# Output das imagens 100 x 100 dos núcleos
output_folder = './image_nucleus'  

# Carregar a imagem em escala de cinza
image_path = ''  # Variável global

def set_image_path(path):
    global image_path  
    image_path = path
    
# Recorta uma imagem 100 x 100 a partir das coordenadas centrais
def cut_image(x, y, size=100):
    with Image.open(image_path) as img:
        left = x - size // 2
        top = y - size // 2
        right = x + size // 2
        bottom = y + size // 2
        img_cropped = img.crop((left, top, right, bottom))
        return img_cropped

# Esvazia a pasta com o output contendo os núcleos da imagem
def esvaziar_pasta(caminho_pasta):
    for item in os.listdir(caminho_pasta):
        caminho_item = os.path.join(caminho_pasta, item)
        if os.path.isfile(caminho_item) or os.path.islink(caminho_item):
            os.unlink(caminho_item)
        elif os.path.isdir(caminho_item):
            shutil.rmtree(caminho_item)
         

def get_characteristics(file_path):
    esvaziar_pasta(output_folder)
    set_image_path(file_path)
    file_name = os.path.basename(file_path)
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    linhas_filtradas = df[df['image_filename'] == file_name]  # Linhas da imagem no csv
                      
    characteristics_list = []  # Lista para armazenar os valores

    # Passando por todos os núcleos daquela imagem
    for index, row in linhas_filtradas.iterrows():
        actual_class = row['bethesda_system']
        cropped_image = cut_image(row['nucleus_x'], row['nucleus_y'])
        output_path = os.path.join(output_folder, f'{index}.png')
        cropped_image.save(output_path)
        cropped_image_np = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
                          
        # Aplicar a limiarização
        centro_y, centro_x = cropped_image_np.shape[0] // 2, cropped_image_np.shape[1] // 2
        blurred_image = cv2.GaussianBlur(cropped_image_np, (5, 5), 0)
        # limiar baseado no ponto central
        valor_pixel_central = blurred_image[centro_y, centro_x] * 1.37
        _, img_thresholded = cv2.threshold(
            blurred_image, valor_pixel_central, 255, cv2.THRESH_BINARY)
        
        # Aplicar o detector de bordas Sobel
        sobelx = cv2.Sobel(img_thresholded, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_thresholded, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobelx, sobely)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(sobel.astype(
            'uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        centro_x, centro_y = cropped_image_np.shape[1] // 2, cropped_image_np.shape[0] // 2
        dist_minima = float('inf')
        contorno_central = None
        # Encontrando contorno mais próximo do centro
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                dist = np.sqrt((centro_x - cx) ** 2 + (centro_y - cy) ** 2)
                if dist < dist_minima:
                    dist_minima = dist
                    contorno_central = contour

        # Extraindo dados a partir do contorno central
        area = 0
        perimetro = 0
        excentricidade = 0
        compacidade = 0
        
        if contorno_central is not None:
            cv2.drawContours(cropped_image_np, [contorno_central], -1, (0, 255, 0), 2)
            area = cv2.contourArea(contorno_central)
            perimetro = cv2.arcLength(contorno_central, True)

            # Calculando a compacidade
            compacidade = (perimetro ** 2) / (4 * np.pi * area) if area != 0 else 0

            # Calculando a circularidade
            circularidade = (4 * np.pi * area) / (perimetro ** 2) if perimetro != 0 else 0

            # Calculando a excentricidade
            if len(contorno_central) >= 5:  # Necessário para ajustar uma elipse
                (x, y), (eixo_menor, eixo_maior), angle = cv2.fitEllipse(contorno_central)
                excentricidade = np.sqrt(1 - (eixo_menor / eixo_maior) ** 2) if eixo_maior != 0 else 0
            else:
                excentricidade = 0

            cv2.circle(cropped_image_np, (centro_x, centro_y), 2, (255, 0, 0), -1)
            #print(f'Área: {area}, Perímetro: {perimetro}, Compacidade: {compacidade}, Circularidade: {circularidade}, Excentricidade: {excentricidade}, Classe: {actual_class}')

        # Adicione os valores à lista
        characteristics_list.append((index, excentricidade, area, compacidade, actual_class))
                
        #cv2.imshow('Imagem com Centro de Contorno Central', cropped_image_np)

        # Exibir a imagem
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Retorna a lista de características
    return characteristics_list

#get_characteristics('./cell_images/1f8274db391ed2747c0c3a8560bf6d3a.png')


