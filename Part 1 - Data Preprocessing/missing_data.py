# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 20:55:15 2023

@author: josue
"""

# Plantilla de preprocesado de datos - Datos faltantes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
#Importar el data set
dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values  

# eliminando NAs
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
