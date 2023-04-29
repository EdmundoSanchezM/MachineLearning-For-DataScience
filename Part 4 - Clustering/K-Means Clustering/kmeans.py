# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:48:15 2023

@author: josue
"""

# K-Means

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')

# matriz. Obteniendo valores del data frame [filas,columnas]
X = dataset.iloc[:, [3,4]].values #Matriz

# Metodo del codo para averiguar el numero optimo de clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,
                    init="k-means++",
                    max_iter= 300,
                    n_init = 10,
                    random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title("Metodo de codo")
plt.xlabel("Numero de Clusters")
plt.ylabel("WCSS(K)")
plt.show()

#Aplicar el metodo de K-means para segmetar del data set
kmeans = KMeans(n_clusters=5,
                init="k-means++",
                max_iter= 300,
                n_init = 10,
                random_state=0)
y_kmeans = kmeans.fit_predict(X)

#Visualizacion de los cluster
plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1],s=100,c="red", label = "Cautos")
plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1],s=100,c="blue", label = "Estandar")
plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1],s=100,c="green", label = "Objetivo")
plt.scatter(X[y_kmeans == 3,0], X[y_kmeans == 3,1],s=100,c="cyan", label = "Descuidados")
plt.scatter(X[y_kmeans == 4,0], X[y_kmeans == 4,1],s=100,c="magenta", label = "Conservadores")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c="yellow", label = "Baricentros")
plt.title("Cluster de clientes")
plt.xlabel("Sueldo anual en (k $)")
plt.ylabel("Puntuacion en Gastos (1-100)")
plt.legend()
plt.show()