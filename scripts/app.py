from tkinter import CENTER
import customtkinter as ctk
from customtkinter import filedialog
from PIL import Image, ImageTk
import Mahalanobis_binary
import re
import plot_graphs
import os
import shutil
import tempfile

def load_image_button():
    insert_image_btn = ctk.CTkButton(scrollable_frame, text='Carregar Nova Imagem', command=upload_image)
    insert_image_btn.grid(row=0, column=0, pady=(40))

def copy_images_to_temp_folder():
    temp_folder = tempfile.mkdtemp()

    for filename in os.listdir('./image_nucleus'):
        src_path = os.path.join('./image_nucleus', filename)
        dest_path = os.path.join(temp_folder, filename)

        # Copiar a imagem para o diretório temporário
        shutil.copy(src_path, dest_path)

    return temp_folder

def display_nucleus():
    max_column_number = 15

    scrollable_frame.nucleus_frame = ctk.CTkFrame(scrollable_frame)
    scrollable_frame.nucleus_frame.grid(row=2, column=0, padx=10, pady=10)

    nucleus_frame_title = ctk.CTkLabel(scrollable_frame.nucleus_frame, text="Núcleos Identificados")
    nucleus_frame_title.grid(row=0, column=0, pady=20, columnspan=max_column_number)

    row_index = 1
    column_index = 0
    file_index = 1

    temp_folder = copy_images_to_temp_folder()

    for filename in os.listdir(temp_folder):
        if column_index > max_column_number - 1:
            column_index = 0
            row_index += 1

        image_path = os.path.join(temp_folder, filename)
        photo = ctk.CTkImage(light_image=Image.open(image_path),dark_image=Image.open(image_path),size=(50, 50))

        nucleus_image = ctk.CTkLabel(scrollable_frame.nucleus_frame, image=photo, text="")
        nucleus_image.image = photo  # manter uma referência!
        nucleus_image.grid(row=row_index, column=column_index, padx=10, pady=(0, 35))

        nucleus_label = ctk.CTkLabel(scrollable_frame.nucleus_frame, text=file_index)
        nucleus_label.grid(row=row_index, column=column_index, padx=10, pady=(50, 0))

        column_index += 1
        file_index += 1

    return

def display_mehalanobi_results(ai_response):
    true_classes = ai_response['true_classes']
    characteristics_and_classes = ai_response['characteristics_and_classes']
    predicted_classes = ai_response['predicted_classes']
    accuracy = ai_response['accuracy']

    graph_btn = ctk.CTkButton(scrollable_frame, text='Mostrar Gráfico', command=lambda: plot_graphs.plot_graph_mehalanobis_binary(characteristics_and_classes))
    graph_btn.pack()

    confusion_graph_btn = ctk.CTkButton(scrollable_frame, text='Mostrar Matriz de Confusão', command=lambda: plot_graphs.plot_graph_mehalanobis_binary_confusion(true_classes, predicted_classes))
    confusion_graph_btn.pack()

    string = 'Accuracy: ' + "{:.2f}".format(accuracy * 100) + "%"
    label_image = ctk.CTkLabel(scrollable_frame, text=string)
    label_image.pack()   
    return


def upload_image():
    caminho_arquivo = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")]) # abre janela de dialogo

    for widget in scrollable_frame.winfo_children(): # limpar a tela 
        widget.destroy()

    load_image_button() # renderizar botao de carregar imagem novamente

    if caminho_arquivo:
        pattern = r'/([^/]+)$'
        match = re.search(pattern, caminho_arquivo)

        if match:
            result = "./cell_images/"+match.group(1)
            print(result)
      
        ai_response = Mahalanobis_binary.classify_mahalanobis_binary(result)
        
        photo = ctk.CTkImage(light_image=Image.open(caminho_arquivo),dark_image=Image.open(caminho_arquivo),size=(640, 480))

        label_image = ctk.CTkLabel(scrollable_frame, text='', image=photo)
        label_image.image = photo
        label_image.grid(row=1, column=0, pady=(40))  
        
        display_nucleus()
        # display_mehalanobi_results(ai_response)

# Configurações iniciais da janela
width = 999
height = 800

window = ctk.CTk()
window.geometry("1200x800")
window.title('Processo Papanicolau')

# Criação do Scrollable Frame
scrollable_frame = ctk.CTkScrollableFrame(window)
scrollable_frame.pack(fill="both", expand=True)
scrollable_frame.grid_columnconfigure(0, weight=1)
scrollable_frame.grid_rowconfigure(0, weight=1)

# display_nucleus()

insert_image_btn = ctk.CTkButton(scrollable_frame, text='Carregar Imagem', command=upload_image)
insert_image_btn.grid(row=0, column=0, pady=(40))

window.mainloop()
