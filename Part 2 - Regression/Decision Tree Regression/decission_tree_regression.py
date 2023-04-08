# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 01:46:32 2023

@author: josue
"""

# Regresion con Arboles de Decision

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
from sklearn.tree import DecisionTreeRegressor
#A la hora de cortar se usa el error cuadrado medio tipicamente
regression = DecisionTreeRegressor(random_state=0)
regression.fit(X,y)

# Prediccion de nuestros modelos
y_pred = regression.predict([[6.5]])


# Visualización de los resultados del Modelo Polinómico
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color = "red")
plt.plot(X, regression.predict(X), color = "blue")
plt.title("Modelo de Arboles de Decision para Regresión")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()


