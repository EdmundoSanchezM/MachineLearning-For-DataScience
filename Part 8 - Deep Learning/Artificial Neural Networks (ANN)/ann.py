# -*- coding: utf-8 -*-
"""
Created on Tue May  9 21:16:29 2023

@author: josue
"""

# Redes Neuronales Artificales

# Instalar Tensorflow y Keras
# Crear environment conda create -n nombre python=version anaconda
# activate nombre
# conda install spyder
# pip install tensorflow
# pip install keras

# Parte 1 - Pre procesado de datos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Churn_Modelling.csv')


X = dataset.iloc[:, 3:-1].values #Matriz
y = dataset.iloc[:, -1].values  # vector

# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X_Gender = LabelEncoder()
X[:,2] = labelencoder_X_Gender.fit_transform(X[:,2])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],
    remainder='passthrough'
)
X = np.array(ct.fit_transform(X), dtype=float)
#Evitar la trampa de las variables dummy.Podemos eliminar cualquiera
X = X[:, 1:]

# Dividir data set entre conjunto de entranamiento y testing
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2, random_state=0)

# Escalado de variables - Obligatorio
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Parte 2 - Construir la RNA
import keras
from keras.models import Sequential
from keras.layers import Dense
"""
11 NODOS DE ENTRADA. CAPA ENTRADA
6 NODOS EN LA SEGUNDA CAPA. CAPA OCULTA
6 NODOS EN LA TERCERA CAPA. CAPA OCULTA
1 NODO CAPA DE SALIDA
"""
# Iniciar la RNA
# 2 formas. 1.- Definir la secuencia de capas
# 2.- Definir el grafo de como se relacionan las capas
classifier = Sequential()

# Añadir las capas de entrada y primera capa oculta
classifier.add(Dense(units=6, kernel_initializer="uniform", activation = 'relu',
                     input_dim = 11))#Experimentacion

# Añadir la segunda capa oculta. No ocupamos input_dim ya que ya se sabe
classifier.add(Dense(units=6, kernel_initializer="uniform", activation = 'relu')) 

# Añadir la Capa de salida. Sigmoid da probabilidad
classifier.add(Dense(units=1, kernel_initializer="uniform", activation = 'sigmoid')) 

# Compilar la RNA
# binary_crossentropy: Diferencia y aplicar logaritmo al resultado para 
# transformar las categorias a numeros
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Ajustamos la RNA al Conjunto de Entrenamiento
classifier.fit(X_train, y_train, batch_size = 5, epochs = 150)

#Parte 3 - Evaluar el modelo y calcular las predicciones finales
# Prediccion de los resultados con el conjunto de Testing
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5) #1 sale 0 no sale
# Elaborar una matriz de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred,labels=[0,1])

