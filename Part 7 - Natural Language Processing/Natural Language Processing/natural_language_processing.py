# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:02:51 2023

@author: josue
"""

# Natural Lenguage Processing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset              
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t",quoting=3)   

# Limpieza de texto, quitamos conectores y simplificamos verbos
import re
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range (0,len(dataset['Review'])):
    review = re.sub('[^a-zA-Z]', ' ',dataset['Review'][i]).lower().split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    corpus.append(' '.join(review))

# Crear el Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)#n palabras mas frecuentes
X = cv.fit_transform(corpus).toarray()#Matriz dispersa / de caracteristicas
y = dataset.iloc[:, 1].values

# Dividir data set entre conjunto de entranamiento y testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.20, random_state=0)

# Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy",random_state=0)
classifier.fit(X_train, y_train)


# Prediccion de los resultados con el conjunto de Testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusion
from sklearn.metrics import confusion_matrix

#Diagonal nos dice las predicciones correctas, las otras nos dicen que son predicciones incorrectas 
cm = confusion_matrix(y_test, y_pred,labels=[0,1])#Orden de los labels determinanr la diagonal

precision = ((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))
