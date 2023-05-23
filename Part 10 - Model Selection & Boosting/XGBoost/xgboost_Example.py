# -*- coding: utf-8 -*-
"""
Created on Mon May 22 18:18:34 2023

@author: josue
"""

# XGBoost
# Las instrucciones de instalación se pueden consultar en http://xgboost.readthedocs.io/en/latest/build.html

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

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

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Ajustar el modelo XGBoost al Conjunto de Entrenamiento
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Aplicar k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

#Aplicar la mejora de Grid Search para optimizar el modelo y sus parametros
from sklearn.model_selection import GridSearchCV
parameters  = {'max_depth': [3, 5, 7],
               'learning_rate': [0.1, 0.01, 0.001],
               'n_estimators': [100, 500, 1000]}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1) 

grid_search = grid_search.fit(X_train, y_train)
best_accuraacy = grid_search.best_score_
best_parameters = grid_search.best_params_
