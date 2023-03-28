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

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values  # vector

#Dividir data set entre conjunto de entranamiento y testing
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2, random_state=0)

#Escalado de variables
"""sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""



















