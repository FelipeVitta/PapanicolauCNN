import customtkinter as ctk
from customtkinter import filedialog
from PIL import Image
import Mahalanobis_binary
import Mahalanobis_categorical
import Convolutional_binary
import re
import plot_graphs
import nucleus_detection
import os
import shutil
import tempfile
from sklearn.metrics import accuracy_score

# CORES
button_color = "#E6E6E6"
button_hover_color = "#CCCCCC"

button_txt_color = "black"

table_bg_color = "#383838"

negative_txt_color = "#c71f1f"
positive_txt_color = "#2cac1b"

area_txt_color = "#4f80c9"
comp_txt_color = "#4fc974"
exc_txt_color = "#c94f80"

zoom_count = 0

true_categorical_classes = []
true_binary_classes = []

def load_image_button():
    insert_image_btn = ctk.CTkButton(
        scrollable_frame, 
        text='Carregar Nova Imagem', 
        command=upload_image,
        fg_color=button_color,
        text_color=button_txt_color,
        corner_radius=80,
        font=normal_font_small,
        hover_color=button_hover_color)
    insert_image_btn.grid(row=0, column=0, pady=(20))

def copy_images_to_temp_folder():
    temp_folder = tempfile.mkdtemp()

    for filename in os.listdir('./image_nucleus'):
        src_path = os.path.join('./image_nucleus', filename)
        dest_path = os.path.join(temp_folder, filename)

        shutil.copy(src_path, dest_path)

    return temp_folder

def display_nucleus(nucleus_infos):
    max_column_number = 10
    # container geral
    nucleus_frame = ctk.CTkFrame(scrollable_frame)
    nucleus_frame.grid(row=3, column=0, padx=10, pady=10)

    # linha do título
    nucleus_frame_title = ctk.CTkLabel(nucleus_frame, text="Núcleos Identificados", font=title_font)
    nucleus_frame_title.grid(row=0, column=0, pady=(20, 10), columnspan=max_column_number)

    # container legenda
    nucleus_frame.label = ctk.CTkFrame(nucleus_frame, fg_color="transparent")
    nucleus_frame.label.grid(row=1, column=0, pady=(10, 20), columnspan=max_column_number)

    # legenda
    area_label_circle = ctk.CTkLabel(nucleus_frame.label, text="", fg_color=area_txt_color, height=8, width=15, corner_radius=80)
    area_label_circle.grid(row=1, column=0, padx=10)

    area_label_txt = ctk.CTkLabel(nucleus_frame.label, text="Área", text_color=area_txt_color, font=normal_font_bold)
    area_label_txt.grid(row=1, column=1, padx=(0, 20))

    exc_label_circle = ctk.CTkLabel(nucleus_frame.label, text="", fg_color=exc_txt_color, height=8, width=15, corner_radius=80)
    exc_label_circle.grid(row=1, column=2, padx=10)

    exc_label_txt = ctk.CTkLabel(nucleus_frame.label, text="Excentricidade", text_color=exc_txt_color, font=normal_font_bold)
    exc_label_txt.grid(row=1, column=3, padx=(0, 20))

    comp_label_circle = ctk.CTkLabel(nucleus_frame.label, text="", fg_color=comp_txt_color, height=8, width=15, corner_radius=80)
    comp_label_circle.grid(row=1, column=4, padx=10)

    comp_label_txt = ctk.CTkLabel(nucleus_frame.label, text="Compacidade", text_color=comp_txt_color, font=normal_font_bold)
    comp_label_txt.grid(row=1, column=5, padx=(0, 20))

    row_index = 2
    column_index = 0

    temp_folder = copy_images_to_temp_folder()

    file_index = 1
    for filename in os.listdir(temp_folder):
        if column_index > max_column_number - 1:
            column_index = 0
            row_index += 1

        # container da imagem e index do nucleo + descricao
        nucleus_frame.nucleus_info = ctk.CTkFrame(nucleus_frame, fg_color="transparent")
        nucleus_frame.nucleus_info.grid(row=row_index, column=column_index, padx=10, pady=10)

        # container da imagem do nucleo + index do núcleo
        nucleus_frame.nucleus_info.picture = ctk.CTkFrame(nucleus_frame.nucleus_info)
        nucleus_frame.nucleus_info.picture.grid(row=0, column=0, padx=10, pady=0)

        image_path = os.path.join(temp_folder, filename)
        photo = ctk.CTkImage(light_image=Image.open(image_path), dark_image=Image.open(image_path), size=(55, 55))

        nucleus_image = ctk.CTkLabel(nucleus_frame.nucleus_info.picture, image=photo, text="")
        nucleus_image.grid(row=0, column=0, pady=0, padx=0)

        nucleus_label = ctk.CTkLabel(nucleus_frame.nucleus_info.picture, text=file_index, font=normal_font_bold)
        nucleus_label.grid(row=1, column=0, pady=0, padx=0)

        # container das caracteristicas do núcleo
        nucleus_frame.nucleus_info.picture_info = ctk.CTkFrame(nucleus_frame.nucleus_info)
        nucleus_frame.nucleus_info.picture_info.grid(row=0, column=1, padx=0, pady=0, sticky="n")

        characteristics = nucleus_infos[file_index - 1][0]
        area = ctk.CTkLabel(
            nucleus_frame.nucleus_info.picture_info, 
            text="{:.2f}".format(characteristics[0]), 
            font=normal_font_small, 
            height=20, 
            text_color=area_txt_color)
        area.grid(row=0, column=0, padx=0, pady=0, sticky="w")

        excentricidade = ctk.CTkLabel(
            nucleus_frame.nucleus_info.picture_info, 
            text="{:.2f}".format(characteristics[1]), 
            font=normal_font_small, 
            height=20,
            text_color=exc_txt_color)
        excentricidade.grid(row=1, column=0, padx=0, pady=0, sticky="w")

        compacidade = ctk.CTkLabel(
            nucleus_frame.nucleus_info.picture_info, 
            text="{:.2f}".format(characteristics[2]), 
            font=normal_font_small, 
            height=20,
            text_color=comp_txt_color)
        compacidade.grid(row=2, column=0, padx=0, pady=0, sticky="w")

        column_index += 1
        file_index += 1

def render_table(
        frame, title, table_headers, 
        table_data, accuracy, 
        confusion_graph_btn_function, 
        row = 0, column = 0, 
        negative_display_text = "Negativo",
        positive_display_text = "Positivo"):
    frame.results_frame = ctk.CTkFrame(frame, fg_color="transparent")
    frame.results_frame.grid(row=row, column=column, padx=10, pady=10)

    results_frame_title = ctk.CTkLabel(frame.results_frame, text=title, font=title_font)
    results_frame_title.grid(row=0, column=0, pady=10, columnspan=2)

    frame.results_frame.buttons = ctk.CTkFrame(frame.results_frame)
    frame.results_frame.buttons.grid(row=1, column=0, pady=(0, 10), padx=10)

    frame.results_frame.table = ctk.CTkFrame(frame.results_frame, fg_color=table_bg_color)
    frame.results_frame.table.grid(row=2, column=0, padx=10)

    column_count = 0
    for header in table_headers:
        header_label = ctk.CTkLabel(frame.results_frame.table, text=header, font=normal_font_bold, height=30)
        header_label.grid(row=0, column=column_count, padx=10, pady=10)

        row_count = 1
        for data in table_data:
            text_color = "white"
            row_text = data[column_count]

            if row_text == "Negative for intraepithelial lesion":
                row_text = negative_display_text
                text_color = negative_txt_color
            elif row_text == "Positive for intraepithelial lesion":
                row_text = positive_display_text
                text_color = positive_txt_color

            value_label = ctk.CTkLabel(frame.results_frame.table, text=row_text, text_color=text_color, font=normal_font)
            value_label.grid(row=row_count, column=column_count, padx=10)

            row_count += 1  

        column_count += 1

    string = 'Acurácia: ' + "{:.2f}".format(accuracy * 100) + "%"
    accuracy_label = ctk.CTkLabel(frame.results_frame.buttons, text=string, font=normal_font_small)
    accuracy_label.grid(row=1, column=0)

    confusion_graph_btn = ctk.CTkButton(
        frame.results_frame.buttons, 
        text='Matriz de\nConfusão', 
        command=confusion_graph_btn_function,
        fg_color=button_color,
        hover_color=button_hover_color,
        text_color=button_txt_color,
        font=normal_font_small,
        corner_radius=80,
        width=80)
    confusion_graph_btn.grid(row=2, column=0, pady=10, padx=10)

def display_results(frame, predicted_classes, true_classes, title, headers, column=0, binary=False):
    true_classes = true_binary_classes
    accuracy = accuracy_score(true_classes, predicted_classes)

    show_confusion_graph = lambda: plot_graphs.plot_confusion_graph(predicted_classes, true_classes, binary)

    data = []

    nucleus_number = 1
    for predicted_class in predicted_classes:
        table_row = [nucleus_number, predicted_class]
        data.append(table_row)
        nucleus_number += 1

    render_table(
        frame=frame,
        column=column,
        title=title, 
        table_headers=headers, 
        table_data=data, 
        accuracy=accuracy, 
        confusion_graph_btn_function=show_confusion_graph)
    
def zoom_in(image):
    new_size = (int(image._size[0] * 1.2), int(image._size[1] * 1.2))
    global zoom_count
    if(zoom_count < 5):
        zoom_count += 1
        image.configure(size=new_size)

def zoom_out(image):
    new_size = (int(image._size[0] / 1.2), int(image._size[1] / 1.2))
    global zoom_count
    if(zoom_count > -5):
        zoom_count -= 1
        image.configure(size=new_size)

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
      
        scrollable_frame.zoom_buttons_frame = ctk.CTkFrame(scrollable_frame)
        scrollable_frame.zoom_buttons_frame.grid(row=1, column=0, pady=(10, 0))

        zoom_in_button = ctk.CTkButton(
            scrollable_frame.zoom_buttons_frame, 
            text='+', 
            command=lambda: zoom_in(photo), 
            width=30, 
            fg_color=button_color, 
            font=normal_font_small,
            text_color=button_txt_color,
            corner_radius=80,
            hover_color=button_hover_color)
        zoom_in_button.grid(row=0, column=0, padx=10)

        zoom_out_button = ctk.CTkButton(
            scrollable_frame.zoom_buttons_frame, 
            text='-', 
            command=lambda: zoom_out(photo), 
            width=30, 
            fg_color=button_color, 
            font=normal_font_small,
            text_color=button_txt_color,
            corner_radius=80,
            hover_color=button_hover_color)
        zoom_out_button.grid(row=0, column=1, padx=10)

        # exibir imagem escolhida
        photo = ctk.CTkImage(light_image=Image.open(file_path), size=(480, 360))
        label_image = ctk.CTkLabel(scrollable_frame, text='', image=photo)
        label_image.grid(row=2, column=0, pady=(20))  

        mahalanobis_binary_nucleus_info = []
        mahalanobis_nucleus_info = []

        global true_categorical_classes, true_binary_classes
        true_categorical_classes = []
        true_binary_classes = []

        feat = nucleus_detection.get_characteristics(result)
        for characteristic in feat:
            excentricidade = characteristic[1]
            area = characteristic[2]
            compacidade = characteristic[3]
            classe = characteristic[4]

            mahalanobis_nucleus_info.append(([area, excentricidade, compacidade], classe)) 

            true_categorical_classes.append(classe)
            if(classe != 'Negative for intraepithelial lesion'):
                classe = 'Positive for intraepithelial lesion'
            true_binary_classes.append(classe)

            mahalanobis_binary_nucleus_info.append(([area, excentricidade, compacidade], classe)) 

        scrollable_frame.dispersion_buttons_frame = ctk.CTkFrame(scrollable_frame)
        scrollable_frame.dispersion_buttons_frame.grid(row=4, column=0, pady=(10, 0))

        dispersion_graph_categorical = lambda: plot_graphs.plot_dispersion_graph(mahalanobis_nucleus_info)
        dispersion_graph_categorical_btn = ctk.CTkButton(
            scrollable_frame.dispersion_buttons_frame, 
            text='Gráfico de Dispersão\nCategórico', 
            command=dispersion_graph_categorical,
            fg_color=button_color,
            hover_color=button_hover_color,
            font=normal_font_small,
            text_color=button_txt_color,
            corner_radius=80,
            width=80)
        dispersion_graph_categorical_btn.grid(row=0, column=0, pady=10, padx=10)

        dispersion_graph_binary = lambda: plot_graphs.plot_dispersion_graph(mahalanobis_binary_nucleus_info, binary=True)
        dispersion_graph_binary_btn = ctk.CTkButton(
            scrollable_frame.dispersion_buttons_frame, 
            text='Gráfico de Dispersão\nBinário', 
            command=dispersion_graph_binary,
            fg_color=button_color,
            hover_color=button_hover_color,
            font=normal_font_small,
            text_color=button_txt_color,
            corner_radius=80,
            width=80)
        dispersion_graph_binary_btn.grid(row=0, column=1, pady=10, padx=10)

        # exibir tabelas
        binary_headers = ["Núcleo", "Resultado para \n lesão intraepitelial"]
        categorical_headers = ["Núcleo", "Classe Predita"]

        scrollable_frame.display_results_frame = ctk.CTkFrame(scrollable_frame)
        scrollable_frame.display_results_frame.grid(row=5, column=0, padx=10, pady=10)

        mahalanobis_binary_response = Mahalanobis_binary.classify_mahalanobis_binary(mahalanobis_binary_nucleus_info)
        display_results(
            frame=scrollable_frame.display_results_frame,
            predicted_classes=mahalanobis_binary_response, 
            title="Mahalanobis Binário",
            headers=binary_headers,
            true_classes=true_binary_classes,
            binary=True)

        mahalanobis_response = Mahalanobis_categorical.classify_mahalanobis(mahalanobis_nucleus_info)
        display_results(
            frame=scrollable_frame.display_results_frame,
            column=1,
            predicted_classes=mahalanobis_response, 
            title="Mahalanobis Categórico",
            headers=categorical_headers,
            true_classes=true_categorical_classes)

        convolutional_binary_response = Convolutional_binary.classify_convolutional_binary()
        display_results(
            frame=scrollable_frame.display_results_frame,
            column=2,
            predicted_classes=convolutional_binary_response, 
            title="Convolucional Binário",
            headers=binary_headers,
            true_classes=true_binary_classes,
            binary=True)

        # convolutional_response = Convolutional_categorical.classify_convolutional()
        # TODO: TROCAR PARAMETRO convolutional_binary_response PARA convolutional_response DEPOIS
        display_results(
            frame=scrollable_frame.display_results_frame,
            column=3,
            predicted_classes=convolutional_binary_response, 
            title="Convolucional Categórico",
            headers=categorical_headers,
            true_classes=true_categorical_classes)

        display_nucleus(mahalanobis_nucleus_info)


# Configurações iniciais da janela
width = 999
height = 800

window = ctk.CTk()
window.geometry("1500x900")
window.title('Processo Papanicolau')

# Criação do Scrollable Frame
scrollable_frame = ctk.CTkScrollableFrame(window)
scrollable_frame.pack(fill="both", expand=True)
scrollable_frame.grid_columnconfigure(0, weight=1)
scrollable_frame.grid_rowconfigure(0, weight=1)

# FONTES 
title_font = ctk.CTkFont(family="Roboto", size=18, weight="bold")
normal_font_bold = ctk.CTkFont(family="Roboto", size=14, weight="bold")
normal_font_small = ctk.CTkFont(family="Roboto", size=12, weight="bold")
normal_font = ctk.CTkFont(family="Roboto", size=14)

insert_image_btn = ctk.CTkButton(
    scrollable_frame, 
    text='Carregar Imagem', 
    command=upload_image,
    fg_color=button_color,
    font=normal_font_small,
    text_color=button_txt_color,
    corner_radius=80,
    hover_color=button_hover_color)
insert_image_btn.grid(row=0, column=0, pady=(20))

window.mainloop()
