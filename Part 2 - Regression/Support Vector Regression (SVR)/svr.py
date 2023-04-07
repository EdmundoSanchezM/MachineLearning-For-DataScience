# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:13:59 2023

@author: josue
"""

# SVR

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
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1))

# Ajustar la regresion con el data set
from sklearn.svm import SVR
regression = SVR(kernel="rbf")
regression.fit(X,y)


# Prediccion de nuestros modelos
y_pred = regression.predict(sc_X.transform(np.array([[6.5]])))
y_pred = sc_y.inverse_transform(y_pred)

#Eliminando escalado para graficacion
X = sc_X.inverse_transform(X)
y = sc_y.inverse_transform(y)
# Visualización de los resultados del Modelo Polinómico
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid,  sc_y.inverse_transform(regression.predict(sc_X.transform(X_grid))), color = "blue")
plt.title("Modelo de Regresión (SVR)")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()
