# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 23:40:34 2023

@author: josue
"""

# Clustering Jerarquicio Aglomerativo

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')

# matriz. Obteniendo valores del data frame [filas,columnas]
X = dataset.iloc[:, [3,4]].values #Matriz

# Utilizar el dendrograma para encontrar el numero optimo de clusters
import scipy.cluster.hierarchy as sch
dendrograma = sch.dendrogram(sch.linkage(y = X, method="ward"))#Ward minimiza la variancia entre cada punto de los clasters
plt.title("Dendrograma")
plt.xlabel("Clientes")
plt.ylabel("Distancia Euclida")
plt.show()

#Ajustar el clustering jerarquico a nuestro conjunto de datos
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity="euclidean",linkage="ward") 
y_hc = hc.fit_predict(X)

#Visualizacion de los cluster 2D
plt.scatter(X[y_hc == 0,0], X[y_hc == 0,1],s=100,c="red", label = "Cautos")
plt.scatter(X[y_hc == 1,0], X[y_hc == 1,1],s=100,c="blue", label = "Estandar")
plt.scatter(X[y_hc == 2,0], X[y_hc == 2,1],s=100,c="green", label = "Objetivo")
plt.scatter(X[y_hc == 3,0], X[y_hc == 3,1],s=100,c="cyan", label = "Descuidados")
plt.scatter(X[y_hc == 4,0], X[y_hc == 4,1],s=100,c="magenta", label = "Conservadores")
plt.title("Cluster de clientes")
plt.xlabel("Sueldo anual en (k $)")
plt.ylabel("Puntuacion en Gastos (1-100)")
plt.legend()
plt.show()