# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 19:16:37 2023

@author: josue
"""

# Regresion lineal multiple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('50_Startups.csv')

# matriz. Obteniendo valores del data frame [filas,columnas]
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Codificar datos categoricos
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],
    remainder='passthrough'
)
#NumPy con las columnas dummy adicionales creadas
X = np.array(ct.fit_transform(X), dtype=float)

#Evitar la trampa de las variables dummy.Podemos eliminar cualquiera
X = X[:, 1:]

#Dividir data set entre conjunto de entranamiento y testing
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2, random_state=0)

# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

# Ajustar el modelo de regresion lineal multiple con el conjunto de entrenamiento
regression = LinearRegression()
regression.fit(X_train,y_train)#Teoricamente esto es all in

# Prediccion de los resultados en el conjunto de testing
y_pred = regression.predict(X_test)

# Construir el modelo optimo de RLM utilizando Eliminacion hacia atras
import statsmodels.api as sm
#Agregando termino independiente. Para calcular P valor
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)#0 fila 1 columna, al inicio 1's
#Paso 1
SL = 0.05
#Paso 2. Calculando el modelo con todas las VI con statsmodels, usando Ordinary Least Squares
#Mínimos cuadrados ordinarios
#Nueva matriz de caracteristicas solamente con las optimas
X_opt =  X[:, [0,1,2,3,4,5]] #all in
#endog:= variable a predecir (intrinseca), exog:= matriz de caracteristicas
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#Opteniendo P valor (P>|t|), entre mas informacion. Probabilidad de 0 en el intervalo de confianza
regression_OLS.summary()

#Reajustamos modelo
X_opt =  X[:, [0,1,3,4,5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()#Output no corresponde ya, tener cuidado

#Reajustamos modelo
X_opt =  X[:, [0,3,4,5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()#Output no corresponde ya, tener cuidado

#Reajustamos modelo
X_opt =  X[:, [0,3,5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()#Output no corresponde ya, tener cuidado

#Reajustamos modelo
X_opt =  X[:, [0,3]] #Se volvio en LSR
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()#Output no corresponde ya, tener cuidado








