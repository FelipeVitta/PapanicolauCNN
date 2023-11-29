import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

cores = {'Negative for intraepithelial lesion': 'red', 'Positive for intraepithelial lesion': 'blue'}
classes = [] # Lista para armazenar as classes   
areas = []
compacidades = []
# Função para plotar o gráfico 
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

def plot_graph_mehalanobis_binary_confusion(predicted_classes, true_classes):
    confusion_mat = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Classe Prevista')
    plt.ylabel('Classe Verdadeira')
    plt.title('Matriz de Confusão')
    plt.show()