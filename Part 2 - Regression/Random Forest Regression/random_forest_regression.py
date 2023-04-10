# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 21:00:28 2023

@author: josue
"""


# Regresion con Bosques Aleatorios

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

# matriz. Obteniendo valores del data frame [filas,columnas]
X = dataset.iloc[:, 1:2].values #Matriz 
y = dataset.iloc[:, 2].values  # vector


#Dividir data set entre conjunto de entranamiento y testing
#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2, random_state=0)


#Escalado de variables
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# Ajustar la regresion con el dataset
from sklearn.ensemble import RandomForestRegressor
#A la hora de cortar se usa el error cuadrado medio tipicamente
#n_estimators: Numero de arboles
regression = RandomForestRegressor(n_estimators = 330, random_state = 0)
regression.fit(X,y)

# Prediccion de nuestros modelos
y_pred = regression.predict([[6.5]])


# Visualización de los resultados del Bosque Aleatorio
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, regression.predict(X_grid), color = "blue")
plt.title("Modelo de Bosques Aleatorios para Regresión")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()


