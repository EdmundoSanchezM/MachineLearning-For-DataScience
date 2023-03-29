# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 19:39:27 2023

@author: josue
"""

#Regresion Lineal Simple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('Salary_Data.csv')

# matriz. Obteniendo valores del data frame [filas,columnas]
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Dividir data set entre conjunto de entranamiento y testing
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 1/3, random_state=0)

# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

#Creacion modelo de Regresion Lineal Simple - Metodo de los minimos cuadrados
regression = LinearRegression()
regression.fit(X_train, y_train)#Mismo tamaño
print("Constante/Intercepto:", regression.intercept_)
print("Pendiente:", regression.coef_)
#Predecir el conjunto de test
y_pred = regression.predict(X_test)

#Visualizar los resultados de entrenamiento
plt.scatter(X_train,y_train,color="red")#Nube de puntos
plt.plot(X_train, regression.predict(X_train))#Recta de regresion
plt.title("Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo (en $)")

#Visualizar los resultados de testing
plt.scatter(X_test,y_test,color="purple")#Nube de puntos
plt.plot(X_train, regression.predict(X_train))#Recta de regresion
plt.title("Sueldo vs Años de Experiencia (Conjunto de Testing)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo (en $)")
