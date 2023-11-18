from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import gdown
import utils as u
base_link = "https://drive.google.com/uc?id="


def create_graph(x, y, labelX, labelY, tipoGrafico: str, title=None, saveFile=False):
    # fig = plt.figure(3, figsize=((20, 5)))
    x = np.array(x[labelX])
    y = np.array(y[labelY])
    fig, ax = plt.subplots()  # a figure with a single Axes
    if title is None:
        title = f"Grafico de {labelX} por {labelY}"
    # Criar o gráfico de linha
    print(min(y), max(y))

    print(min(x), max(x))
    print(tipoGrafico)
    ax.set_ylim(min(y), max(y))
    ax.set_xlim(min(x), max(x))
    ax.set_xlabel(labelX)
    ax.set_ylabel(labelY)
    ax.set_title(title)
    if tipoGrafico == 'bar':
        indices_zero = np.where(y == 0)[0]
        x = np.delete(x, indices_zero)
        y = np.delete(y, indices_zero)
        ax.bar(x, y, width=30, label=labelX)
    if tipoGrafico == 'plot':
        ax.plot(x, y, label=labelX)
    if tipoGrafico == 'scatter':
        ax.scatter(x, y, label=labelX)
    ax.legend()
    # Exibir o gráfico
    if saveFile:
        string_funcao = str(tipoGrafico)
        match = re.search(r"<function (\w+)", string_funcao)
        if match:
            nome_funcao = match.group(1)
        else:
            nome_funcao = tipoGrafico
        fig.savefig(f'graph/{nome_funcao}_{title}.png')

    fig.show()
    fig.clear()
    plt.close(fig)

def create_graph_single_data(y, labelY, tipoGrafico: str, title=None, saveFile=False):
    y = np.array(y[labelY])
    
    if y.size <=0 : 
        return
    x = np.arange(y.size)
    if title is None:
        title = f"Grafico de {labelY}"
    fig, ax = plt.subplots()  # a figure with a single Axes
    
    # Criar o gráfico de linha




    ax.set_ylim(min(y), max(y))
    ax.set_xlim(min(x), max(x))
    ax.set_xlabel('numero da instancia')
    ax.set_ylabel(labelY)
    ax.set_title(title)
    if tipoGrafico == 'bar':
        indices_zero = np.where(x == 0)[0]
        x = np.delete(x, indices_zero)
        y = np.delete(y, indices_zero)
        ax.bar(x, y, width=30, label=labelY)
    if tipoGrafico == 'plot':
        ax.plot(x, y, label=labelY)
    if tipoGrafico == 'scatter':
        ax.scatter(x, y, label=labelY)
    ax.legend()
    # Exibir o gráfico
    if saveFile:
        string_funcao = str(tipoGrafico)
        match = re.search(r"<function (\w+)", string_funcao)
        if match:
            nome_funcao = match.group(1)
        else:
            nome_funcao = tipoGrafico
        fig.savefig(f'graph/{nome_funcao}_{title}.png')

    fig.show()
    fig.clear()
    plt.close(fig)
    
    


def box_plot(x, title, saveFile=False):
    fig, ax = plt.subplots()  # a figure with a single Axes

    ax.boxplot(x[title])
    ax.set_xlabel(title)
    # ax.legend()
    ax.set_title(f'Gráfico de caixa {title}')
    fig.savefig(f'graph/box_plot_{title}.png')
    plt.close(fig)


def main():
    try:
        x = pd.read_csv("files/miplib2017-selected.csv")
        y = pd.read_csv("files/results.csv")
    except Exception as e:
        if not os.path.exists('files'):
            os.mkdir('files')
        print("Trying make download of files")
        miplib = base_link+'14gxX3UyL3b0namBDiErltsnvYCmBcXMU'
        results = base_link+'1yQJ-z7R5MzlexyOjvDe7g97-RYdWZQa2'
        gdown.download(miplib, "files/miplib2017-selected.csv", quiet=True)
        gdown.download(results, "files/results.csv", quiet=True)
        x = pd.read_csv("files/miplib2017-selected.csv")
        y = pd.read_csv("files/results.csv")
    if not os.path.exists("graph"):
        try:
            os.mkdir('graph')
        except:
            print("Could not create graph directory")
    key_r = y.keys()
    key_d = x.keys()
    merged = pd.merge(x, y, left_on='NAME',
                      right_on='instance', how='inner')
    x = merged[list(key_d)]
    y = merged[list(key_r)]
    dropsx = ["NAME", "STATUS", "OBJECTIVE"]
    dropsy = ["instance", 'seed', 'opt.status']
    x = x.drop(dropsx, axis=1)
    y = y.drop(dropsy, axis=1)

    paramsx = [
        x.keys(),
        ['plot',   'scatter', 'bar']
    ]
    paramsy = [
        y.keys(),
        ['plot',   'scatter', 'bar']
    ]
    try:
        print("create_graph")
        for i in list(product(*paramsx)):
            create_graph_single_data(x, *i, saveFile=True)
        # print("box_plot")
        for i in list(product(*paramsy)):
            create_graph_single_data(y, *i, saveFile=True)
    except Exception as e:
        print(f"Error {e}")
        exit()
    for i in list(x.keys()):
        box_plot(x, i, saveFile=True)
    for i in list(y.keys()):
        box_plot(y, i, saveFile=True)


if __name__ == "__main__":
    main()
