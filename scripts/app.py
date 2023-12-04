import customtkinter as ctk
from customtkinter import filedialog
from PIL import Image, ImageTk
import Mahalanobis_binary
import Mahalanobis_categorical
import re
import plot_graphs
import os
import shutil
import tempfile

# CORES
button_color = "#651377"
button_hover_color = "#7F2D91"

table_bg_color = "#383838"

negative_txt_color = "#c71f1f"
positive_txt_color = "#2cac1b"

def load_image_button():
    insert_image_btn = ctk.CTkButton(
        scrollable_frame, 
        text='Carregar Nova Imagem', 
        command=upload_image,
        fg_color=button_color,
        hover_color=button_hover_color)
    insert_image_btn.grid(row=0, column=0, pady=(20))

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

def display_results(
        frame, title, table_headers, 
        table_data, accuracy, graph_btn_function, 
        confusion_graph_btn_function, 
        row = 0, column = 0, 
        negative_display_text = "Negativo",
        positive_display_text = "Positivo"):
    frame.results_frame = ctk.CTkFrame(frame, fg_color="transparent")
    frame.results_frame.grid(row=row, column=column, padx=10, pady=10)

    results_frame_title = ctk.CTkLabel(frame.results_frame, text=title, font=title_font)
    results_frame_title.grid(row=0, column=0, pady=10, columnspan=2)

    frame.results_frame.buttons = ctk.CTkFrame(frame.results_frame)
    frame.results_frame.buttons.grid(row=1, column=0, pady=10, padx=10)

    frame.results_frame.table = ctk.CTkFrame(frame.results_frame, fg_color=table_bg_color)
    frame.results_frame.table.grid(row=2, column=0, padx=10)

    column_count = 0
    for header in table_headers:
        header_label = ctk.CTkLabel(frame.results_frame.table, text=header, font=normal_font_bold)
        header_label.grid(row=0, column=column_count, padx=10, pady=10)

        row_count = 1
        for data in table_data:
            text_color = "white"

            if column_count == 0:
                row_text = row_count # caso seja a primeira coluna, o valor da linha deve ser o numero do nucleo
            else:
                row_text = data[column_count - 1]
                if column_count != len(table_headers) - 1:
                    row_text = "{:.2f}".format(row_text)

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
    accuracy_label = ctk.CTkLabel(frame.results_frame.buttons, text=string, font=normal_font_bold)
    accuracy_label.grid(row=1, column=0, columnspan=2)

    graph_btn = ctk.CTkButton(
        frame.results_frame.buttons, 
        text='Gráfico de Dispersão', 
        command=graph_btn_function,
        fg_color=button_color,
        hover_color=button_hover_color)
    graph_btn.grid(row=2, column=0, pady=10, padx=10)

    confusion_graph_btn = ctk.CTkButton(
        frame.results_frame.buttons, 
        text='Matriz de Confusão', 
        command=confusion_graph_btn_function,
        fg_color=button_color,
        hover_color=button_hover_color)
    confusion_graph_btn.grid(row=2, column=1, pady=10, padx=10)

def display_mahalanobis_binary_results(ai_response, frame):
    true_classes = ai_response['true_classes']
    characteristics_and_classes = ai_response['characteristics_and_classes']
    characteristics_and_predicted_classes = ai_response['characteristics_and_predicted_classes']
    predicted_classes = ai_response['predicted_classes']
    accuracy = ai_response['accuracy']

    show_graph = lambda: plot_graphs.plot_graph_mahalanobis_binary(characteristics_and_classes)
    show_confusion_graph = lambda: plot_graphs.plot_graph_mahalanobis_binary_confusion(true_classes, predicted_classes)
    headers = ["Núcleo", "Área", "Excentricidade", "Compacidade", "Resultado para \n lesão intraepitelial"]
    
    display_results(
        frame=frame,
        column=0,
        title="Mahalanobis Binária", 
        table_headers=headers, 
        table_data=characteristics_and_predicted_classes, 
        accuracy=accuracy, 
        graph_btn_function=show_graph, 
        confusion_graph_btn_function=show_confusion_graph)
    
    
def display_mahalanobis_results(ai_response, frame):
    true_classes = ai_response['true_classes']
    characteristics_and_classes = ai_response['characteristics_and_classes']
    characteristics_and_predicted_classes = ai_response['characteristics_and_predicted_classes']
    predicted_classes = ai_response['predicted_classes']
    accuracy = ai_response['accuracy']

    show_graph = lambda: plot_graphs.plot_graph_mahalanobis(characteristics_and_classes)
    show_confusion_graph = lambda: plot_graphs.plot_graph_mahalanobis_confusion(true_classes, predicted_classes)
    headers = ["Núcleo", "Área", "Excentricidade", "Compacidade", "Classe Predita"]

    display_results(
        frame=frame,
        column=1,
        title="Mahalanobis", 
        table_headers=headers, 
        table_data=characteristics_and_predicted_classes, 
        accuracy=accuracy, 
        graph_btn_function=show_graph, 
        confusion_graph_btn_function=show_confusion_graph,
        negative_display_text="Negativo p/ LI",
        positive_display_text="Positivo p/ LI")
   
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
        
        scrollable_frame.display_results_frame = ctk.CTkFrame(scrollable_frame)
        scrollable_frame.display_results_frame.grid(row=3, column=0, padx=10, pady=10)

        mahalanobis_binary_response = Mahalanobis_binary.classify_mahalanobis_binary(result)
        display_mahalanobis_binary_results(mahalanobis_binary_response, scrollable_frame.display_results_frame)

        mahalanobis_response = Mahalanobis_categorical.classify_mahalanobis(result)
        display_mahalanobis_results(mahalanobis_response, scrollable_frame.display_results_frame)

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

insert_image_btn = ctk.CTkButton(
    scrollable_frame, 
    text='Carregar Imagem', 
    command=upload_image,
    fg_color=button_color,
    hover_color=button_hover_color)
insert_image_btn.grid(row=0, column=0, pady=(20))

window.mainloop()
