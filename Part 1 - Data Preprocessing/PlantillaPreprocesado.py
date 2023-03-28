# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 17:15:54 2023

@author: josue
"""

# Plantilla de preprocesado de datos

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Data.csv')

# matriz. Obteniendo valores del data frame [filas,columnas]
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values  # vector

# eliminando NAs
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(X[:, 1:3])  # [filas,columnas]. Ajustar estos conjuntos
# devuelve y sustitulle eliminando NAs
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Codificar datos categoricos

# Se define el objeto ct de la clase ColumnTransformer
#   El primer argumento de ColumnTransformer es una lista que contiene una tupla
#   con tres elementos: 
        # - 'one_hot_encoder': un nombre que se le da a esta transformación
        # - OneHotEncoder(categories='auto'): una instancia de la clase OneHotEncoder, 
        # que convierte una columna categórica en varias columnas binarias 
        # (dummy) de forma one-hot. categories='auto' indica que las 
        # categorías se detectan automáticamente a partir de los datos de entrada.
        # - [0]: una lista que indica el índice de la columna a transformar.
#`remainder='passthrough' indica que todas las columnas no especificadas en la 
# lista anterior se mantienen sin cambios.
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],
    remainder='passthrough'
)

#Los datos de entrada X se transforman usando ct.fit_transform(X). 
#Esto significa que se aplica la transformación definida en ct a los datos de entrada X. 
#El resultado es una matriz NumPy con las columnas dummy adicionales creadas
X = np.array(ct.fit_transform(X), dtype=float)

labelencoder_Y = preprocessing.LabelEncoder()
# Se codifica automaticamente los valores de 'y' y se sobreescribe.
# Util para cuando tenemos dos valores solamente
y = labelencoder_Y.fit_transform(y)

#Dividir data set entre conjunto de entranamiento y testing
#x entrenamiento y prueba, y de entrenamiento y prueba
#Argumentos(Matriz de datos, vector de datos a predecir, test_size: tamaño del conjunto de test,
#random_state: semilla para generar los arrays)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2, random_state=0)

#Escalado de variables

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



















