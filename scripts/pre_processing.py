import csv
from PIL import Image
import os

# PRÉ PROCESSANDO IMAGENS DE TREINAMENTO E DE TESTES DE CLASSIFICAÇÃO

path_file = 'classifications.csv'
images_folder = './cell_images'
output_folder = './output_cell_images'

# Recorta uma imagem 100 x 100 a partir das cordenadas centrais
def cut_image(image_path, cell_id, x, y, size=100):
    with Image.open(image_path) as img:
        left = x - size // 2
        top = y - size // 2  
        right = x + size // 2
        bottom = y + size // 2
        img_cropped = img.crop((left, top, right, bottom))
        return img_cropped
    
# Se ainda não existir a pasta de output das imagens, então criar    
if not os.path.exists(output_folder):
    os.makedirs(output_folder)  

with open(path_file, newline='', encoding='utf-8') as arquivo_csv:
    leitor_csv = csv.DictReader(arquivo_csv)
    
    for linha in leitor_csv:
        image_id = linha['image_id']
        cell_id = linha['cell_id']
        bethesda_class = linha['bethesda_system']
        x = int(linha['nucleus_x'])
        y = int(linha['nucleus_y'])
        image_filename = linha['image_filename']
        image_path = os.path.join(images_folder, image_filename)

        # Recortar a imagem
        cropped_image = cut_image(image_path, cell_id, x, y)

        # Criar subdiretório para a classe se ele não existir
        class_folder = os.path.join(output_folder, bethesda_class)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

        # Salvar a imagem recortada
        output_path = os.path.join(class_folder, f'{cell_id}.png')
        cropped_image.save(output_path)

print("Processamento concluído.")