import sys

import pandas as pd

import numpy as np

import random


def euclidiana(x, y):
    return np.sqrt(np.power(x - y, 2).sum())


def manhattan(x, y):
    return np.abs(x - y).sum()


def suprema(x, y):
    return np.abs(x - y).max()


def acharclasse(id, df):
    return df.iloc[id][-1]


def knn(k, d, db):
    df = pd.read_csv(db, delim_whitespace=True)
    ds = df.iloc[:, 1: -1]
    ds = ds.to_numpy()

    if d == 0:
        dist = suprema
    elif d == 1:
        dist = manhattan
    else:
        dist = euclidiana
    distancias = []
    distancias2 = []

    for instancia in ds:
        for instanciavizinha in ds:
            distancias.append(
                dist(np.array(instancia), np.array(instanciavizinha)))
        distancias2.append(np.array(distancias))
        distancias.clear()
        # break

    nomedasclasses = df.iloc[:, -1].unique()
    kvizinhos = np.argsort(distancias2)
    kvizinhos = kvizinhos[:, :k+1]

    for index, (kvizinho, dist_amostra) in enumerate(zip(kvizinhos, distancias2)):
        print('Id: ', index + 1)

        count = np.zeros_like(nomedasclasses)

        for id in kvizinho:
            if dist_amostra[id] == 0:
                continue
            else:
                for i, classe in enumerate(nomedasclasses):
                    if acharclasse(id, df) == classe:
                        count[i] += 1
        print('A classe mais provável é: ', nomedasclasses[np.argmax(count)])
        print('Vizinhos mais próximos: ', kvizinho[1:])


def k_means(k, d, aleatorio, db):
    df = pd.read_csv(db, delim_whitespace=True)
    ds = df.iloc[:, 1: -1]
    ds = ds.to_numpy()
    random.seed(aleatorio)
    centroides = []

    if d == 0:
        fun_dist = suprema
    elif d == 1:
        fun_dist = manhattan
    else:
        fun_dist = euclidiana

    for i in range(k):
        centroides.append(random.choice(ds))

    somatoria = []
    count = []  # contador de instâncias do grupo g[i]

    for epoca in range(201):

        for i in centroides:
            somatoria.append([np.zeros_like(ds[0])])
            count.append(0)

        for instancia in ds:
            distancias = []
            for centroide in centroides:
                distancias.append(fun_dist(instancia, centroide))

            somatoria[np.argmin(distancias)] += instancia
            count[np.argmin(distancias)] += 1


        # calculo dos novos centroides:
        for i in range(len(centroides)):
            centroides[i] = somatoria[i] / count[i]


    # classificação com os centroides treinados
    classes = []
    for instancia in ds:

        distancias = []
        for centroide in centroides:
            distancias.append(fun_dist(instancia, centroide))
        classe_label = 'g' + str(np.argmin(distancias)+1)
        classes.append(classe_label)

    coluna_classe = {'class': classes}
    coluna_classe = pd.DataFrame(coluna_classe)
    df['class'] = coluna_classe

    # salvar como txt
    df.to_csv('saidas.txt', sep=' ', index=False, header=True)


def main():
    parametros = sys.argv
    k = int(parametros[3])
    d = int(parametros[5])
    db = parametros[-1]
    aleatorio = int(parametros[7]) if parametros[1] == '--kmeans' else 0
    if parametros[1] == '--knn':
        knn(k, d, db)
    elif parametros[1] == '--kmeans':
        k_means(k, d, aleatorio, db)


if __name__ == "__main__":
    main()
