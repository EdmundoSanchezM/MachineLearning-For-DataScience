# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 20:55:12 2023

@author: josue
"""


# Plantilla de preprocesado de datos - Datos categoricos

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

#Importar el data set
dataset = pd.read_csv('Data.csv')

#Obteniendo valores del data frame
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values  

# Codificar datos categoricos
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],
    remainder='passthrough'
)
X = np.array(ct.fit_transform(X), dtype=float)
labelencoder_Y = preprocessing.LabelEncoder()
y = labelencoder_Y.fit_transform(y)



















