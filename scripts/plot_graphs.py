import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

cores = {'Negative for intraepithelial lesion': 'black', 'Positive for intraepithelial lesion': 'blue'}
classes = [] # Lista para armazenar as classes   
areas = []
compacidades = []

# Mehalanobis binário - plotar o gráfico de dispersão
def plot_graph_mehalanobis_binary(characteristics_and_classes):
    for carac, classe in characteristics_and_classes:
        classes.append('Positive for intraepithelial lesion' if classe != 'Negative for intraepithelial lesion' else classe)   
        areas.append(carac[0])
        compacidades.append(carac[2])
    plt.figure(figsize=(8, 6))
    for classe in set(classes):  # Usar set para obter classes únicas
        indices = [i for i, x in enumerate(classes) if x == classe]
        plt.scatter([areas[i] for i in indices], [compacidades[i] for i in indices], color=cores[classe], label=classe)

    # Adicionar legendas, rótulos e título
    plt.legend()
    plt.xlabel('Área')
    plt.ylabel('Compacidade')
    plt.title('Gráfico de Dispersão por Classe')
    plt.show()
    
# Mehalanobis binário - plotar matriz de confusão
def plot_graph_mehalanobis_binary_confusion(predicted_classes, true_classes):
    confusion_mat = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Classe Verdadeira')
    plt.ylabel('Classe Prevista')
    plt.title('Matriz de Confusão')
    plt.show()
    
cores2 = {'Negative for intraepithelial lesion': 'black', 'ASC-H': 'blue', 'ASC-US':'yellow', 'HSIL':'purple', 'LSIL':'gray', 'SCC':'pink'}
   
# Mehalanobis - plotar o gráfico de dispersão
def plot_graph_mehalanobis(characteristics_and_classes):
    for carac, classe in characteristics_and_classes:
        classes.append(classe) 
        areas.append(carac[0])
        compacidades.append(carac[2])
    plt.figure(figsize=(8, 6))
    for classe in set(classes):  # Usar set para obter classes únicas
        indices = [i for i, x in enumerate(classes) if x == classe]
        plt.scatter([areas[i] for i in indices], [compacidades[i] for i in indices], color=cores2[classe], label=classe)

    # Adicionar legendas, rótulos e título
    plt.legend()
    plt.xlabel('Área')
    plt.ylabel('Compacidade')
    plt.title('Gráfico de Dispersão por Classe')
    plt.show()
    
 # Mehalanobis - plotar matriz de confusão   
def plot_graph_mehalanobis_confusion(predicted_classes, true_classes):
    confusion_mat = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Classe Prevista')
    plt.ylabel('Classe Verdadeira')
    plt.title('Matriz de Confusão')
    plt.show()