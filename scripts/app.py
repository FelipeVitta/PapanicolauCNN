from tkinter import CENTER
import customtkinter as ctk
from customtkinter import filedialog
from PIL import Image, ImageTk
import Mahalanobis_binary
import re
import os

def load_and_display(scrollable_frame):
    for filename in os.listdir('./image_nucleus'):
        image_path = os.path.join('./image_nucleus', filename)
        image = Image.open(image_path)
        image = image.resize((100, 100), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)

        label = ctk.CTkLabel(scrollable_frame, image=photo)
        label.image = photo  # keep a reference!
        label.pack(side='left', padx=10, pady=40)


def upload_image():
    caminho_arquivo = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")]) #abre janela de dialogo
    if caminho_arquivo:
        pattern = r'/([^/]+)$'
        match = re.search(pattern, caminho_arquivo)

        if match:
            result = "./cell_images/"+match.group(1)
            print(result)
      
        Mahalanobis_binary.classify_mahalanobis_binary(result)
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

label_image = ctk.CTkLabel(scrollable_frame, text='Carregue sua imagem do exame Papanicolau aqui:', justify='left')
label_image.pack(pady=(30,0))

window.mainloop()
