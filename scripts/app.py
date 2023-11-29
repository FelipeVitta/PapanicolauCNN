from tkinter import CENTER
import customtkinter as ctk
from customtkinter import filedialog
from PIL import Image, ImageTk
import Mahalanobis_binary
import re
import plot_graphs
import os

def load_and_display(scrollable_frame):
    for filename in os.listdir('./image_nucleus'):
        image_path = os.path.join('./image_nucleus', filename)
        image = Image.open(image_path)
        image = image.resize((50, 50), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)

        label = ctk.CTkLabel(scrollable_frame, image=photo)
        label.image = photo  # keep a reference!
        label.pack(side='left', padx=10, pady=40)


def show_graph(characteristics_and_classes):
    plot_graphs.plot_graph_mehalanobis_binary(characteristics_and_classes)

def show_matrix(true_classes, predicted_classes):
    plot_graphs.plot_graph_mehalanobis_binary_confusion(true_classes, predicted_classes)


def upload_image():
    caminho_arquivo = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")]) #abre janela de dialogo
    if caminho_arquivo:
        pattern = r'/([^/]+)$'
        match = re.search(pattern, caminho_arquivo)

        if match:
            result = "./cell_images/"+match.group(1)
            print(result)
      
        resultado = Mahalanobis_binary.classify_mahalanobis_binary(result)

        true_classes = resultado['true_classes']
        characteristics_and_classes = resultado['characteristics_and_classes']
        predicted_classes = resultado['predicted_classes']
        accuracy = resultado['accuracy']

        graph_btn = ctk.CTkButton(scrollable_frame, text='Mostrar Gráfico', command=lambda: show_graph(characteristics_and_classes))
        graph_btn.pack(pady=(50,0))

        confusion_graph_btn = ctk.CTkButton(scrollable_frame, text='Mostra Matriz de Confusão', command=lambda: show_matrix(true_classes, predicted_classes))
        confusion_graph_btn.pack(pady=(50,0))

        string = 'Accuracy: ' + "{:.2f}".format(accuracy * 100) + "%"
        label_image = ctk.CTkLabel(scrollable_frame, text=string)
        label_image.pack(pady=(30,0))   


    # print(f"Acurácia: {accuracy * 100:.2f}%")
        imagem = Image.open(caminho_arquivo)
        imagem = imagem.resize((640, 480), Image.Resampling.LANCZOS)
        foto = ImageTk.PhotoImage(imagem)

        label_image.configure(image=foto)
        label_image.image = foto
        load_and_display(scrollable_frame)

# Configurações iniciais da janela
width = 1200
height = 800

window = ctk.CTk()
window.geometry(f'{width}x{height}')
window.title('Processo Papanicolau')

# Criação do Scrollable Frame
scrollable_frame = ctk.CTkScrollableFrame(window)
scrollable_frame.pack(fill="both", expand=True)

# Adicionar widgets ao frame interno do CTkScrollableFrame
insert_image_btn = ctk.CTkButton(scrollable_frame, text='Carregar imagem', command=upload_image)
insert_image_btn.pack(pady=(50,0))

label_image = ctk.CTkLabel(scrollable_frame, text='Carregue sua imagem do exame Papanicolau aqui:')
label_image.pack(pady=(30,0))

window.mainloop()
