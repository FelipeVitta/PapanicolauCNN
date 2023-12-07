import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

cores = {'Negative for intraepithelial lesion': 'black', 'Positive for intraepithelial lesion': 'blue'}
cores2 = {'Negative for intraepithelial lesion': 'black', 'ASC-H': 'blue', 'ASC-US':'yellow', 'HSIL':'purple', 'LSIL':'gray', 'SCC':'pink'}
   
# plotar o gráfico de dispersão
def plot_dispersion_graph(characteristics_and_classes, binary=False):
    classes = []
    areas = []
    compacidades = []

    for carac, classe in characteristics_and_classes:
        if binary:
            classes.append('Positive for intraepithelial lesion' if classe != 'Negative for intraepithelial lesion' else classe)  
            title = "Gráfico de Dispersão\nBinário por Classe"
        else:
            classes.append(classe) 
            title = "Gráfico de Dispersão\nCategórico por Classe"

        areas.append(carac[0])
        compacidades.append(carac[2])
    plt.figure(figsize=(8, 6))
    for classe in set(classes):  # Usar set para obter classes únicas
        if binary:
            cor = cores[classe]
        else:
            cor = cores2[classe]

        indices = [i for i, x in enumerate(classes) if x == classe]
        plt.scatter([areas[i] for i in indices], [compacidades[i] for i in indices], color=cor, label=classe)

    # Adicionar legendas, rótulos e título
    plt.legend()
    plt.xlabel('Área')
    plt.ylabel('Compacidade')
    plt.title(title)
    plt.show()
    
# plotar matriz de confusão binário
def plot_confusion_graph(predicted_classes, true_classes, binary=False):
    confusion_mat = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Classe Prevista')
    plt.ylabel('Classe Verdadeira')

    if binary:
        plt.xticks(ticks=[0.5, 1.5], labels=["Negativo", "Positivo"])
        plt.yticks(ticks=[0.5, 1.5], labels=["Negativo", "Positivo"])

    plt.title('Matriz de Confusão')
    plt.show()
