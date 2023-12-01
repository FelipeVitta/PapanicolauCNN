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

        shutil.copy(src_path, dest_path)

    return temp_folder

def display_nucleus():
    max_column_number = 15

    scrollable_frame.nucleus_frame = ctk.CTkFrame(scrollable_frame)
    scrollable_frame.nucleus_frame.grid(row=2, column=0, padx=10, pady=10)

    nucleus_frame_title = ctk.CTkLabel(scrollable_frame.nucleus_frame, text="Núcleos Identificados", font=title_font)
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
        nucleus_image.grid(row=row_index, column=column_index, padx=10, pady=(0, 45))

        nucleus_label = ctk.CTkLabel(scrollable_frame.nucleus_frame, text=file_index, font=normal_font_bold)
        nucleus_label.grid(row=row_index, column=column_index, padx=10, pady=(52, 20))

        column_index += 1
        file_index += 1


def display_mehalanobis_binary_results(ai_response):
    true_classes = ai_response['true_classes']
    characteristics_and_classes = ai_response['characteristics_and_classes']
    characteristics_and_predicted_classes = ai_response['characteristics_and_predicted_classes']
    predicted_classes = ai_response['predicted_classes']
    accuracy = ai_response['accuracy']

    
    scrollable_frame.results_frame = ctk.CTkFrame(scrollable_frame)
    scrollable_frame.results_frame.grid(row=3, column=0, padx=10, pady=10)

    results_frame_title = ctk.CTkLabel(scrollable_frame.results_frame, text="Mehalanobis Binary", font=title_font)
    results_frame_title.grid(row=0, column=0, pady=20, columnspan=2)

    scrollable_frame.results_frame.buttons = ctk.CTkFrame(scrollable_frame.results_frame)
    scrollable_frame.results_frame.buttons.grid(row=1, column=0, padx=10)

    scrollable_frame.results_frame.table = ctk.CTkFrame(scrollable_frame.results_frame)
    scrollable_frame.results_frame.table.grid(row=2, column=0, padx=10)

    headers = ["Núcleo", "Área", "Excentricidade", "Compacidade", "Classe Predita"]

    header_column = 0
    print(characteristics_and_classes)
    for header in headers:
        header_label = ctk.CTkLabel(scrollable_frame.results_frame.table, text=header, font=normal_font_bold)
        header_label.grid(row=0, column=header_column, padx=10)

        value_row = 1


        for nucleus_info in characteristics_and_predicted_classes:
            
            if header_column == 0:
                row_text = value_row # caso seja a primeira coluna, o valor da linha deve ser o numero do nucleo
            else:
                row_text = nucleus_info[header_column - 1]

            value_label = ctk.CTkLabel(scrollable_frame.results_frame.table, text=row_text, font=normal_font)
            value_label.grid(row=value_row, column=header_column, padx=10)
            value_row += 1  

        header_column += 1

    string = 'Accuracy: ' + "{:.2f}".format(accuracy * 100) + "%"
    accuracy_label = ctk.CTkLabel(scrollable_frame.results_frame.buttons, text=string, font=normal_font_bold)
    accuracy_label.grid(row=1, column=0, columnspan=2)

    graph_btn = ctk.CTkButton(
        scrollable_frame.results_frame.buttons, 
        text='Mostrar Gráfico', 
        command=lambda: plot_graphs.plot_graph_mehalanobis_binary(characteristics_and_classes))
    graph_btn.grid(row=2, column=0, pady=10, padx=10)

    confusion_graph_btn = ctk.CTkButton(
        scrollable_frame.results_frame.buttons, 
        text='Mostrar Matriz de Confusão', 
        command=lambda: plot_graphs.plot_graph_mehalanobis_binary_confusion(true_classes, predicted_classes))
    confusion_graph_btn.grid(row=2, column=1, pady=10, padx=10)


    return

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")]) # abre janela de dialogo

    for widget in scrollable_frame.winfo_children(): # limpar a tela 
        widget.destroy()

    load_image_button() # renderizar botao para carregar imagem novamente

    if file_path:
        pattern = r'/([^/]+)$'
        match = re.search(pattern, file_path)

        if match:
            result = "./cell_images/"+match.group(1)
            print(result)
      
        # exibir imagem escolhida
        photo = ctk.CTkImage(light_image=Image.open(file_path), size=(480, 360))
        label_image = ctk.CTkLabel(scrollable_frame, text='', image=photo)
        label_image.grid(row=1, column=0, pady=(20))  
        
        mehalanobis_binary_response = Mahalanobis_binary.classify_mahalanobis_binary(result)
        display_mehalanobis_binary_results(mehalanobis_binary_response)

        display_nucleus()


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

title_font = ctk.CTkFont(family="Roboto", size=18, weight="bold")
normal_font_bold = ctk.CTkFont(family="Roboto", size=14, weight="bold")
normal_font = ctk.CTkFont(family="Roboto", size=14)

# display_nucleus()

insert_image_btn = ctk.CTkButton(scrollable_frame, text='Carregar Imagem', command=upload_image)
insert_image_btn.grid(row=0, column=0, pady=(40))

window.mainloop()
