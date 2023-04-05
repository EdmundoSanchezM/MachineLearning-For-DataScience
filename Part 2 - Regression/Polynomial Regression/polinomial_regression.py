# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:13:59 2023

@author: josue
"""

# Regresion polinomica

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

# matriz. Obteniendo valores del data frame [filas,columnas]
X = dataset.iloc[:, 1:2].values #Matriz 
y = dataset.iloc[:, 2].values  # vector


#Dividir data set entre conjunto de entranamiento y testing

#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2, random_state=0)

# Ajustar la regresion lineal con el data set
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Ajustar la regresion polinomica con el data set
from sklearn.preprocessing import PolynomialFeatures
#Se deben de agregar las x^n hacia la derecha 
poly_reg = PolynomialFeatures(degree = 10)#Por defecto n = 2
X_poly = poly_reg.fit_transform(X)#Salida: termino independiente + b1X1 + b2X1^2 + ... bnX1^n
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

# Visualización de los resultados del Modelo Lineal
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Modelo de Regresión Lineal")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

# Visualización de los resultados del Modelo Polinómico
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title("Modelo de Regresión Polinomica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

# Prediccion de nuestros modelos
lin_reg.predict([[6.5]])
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
