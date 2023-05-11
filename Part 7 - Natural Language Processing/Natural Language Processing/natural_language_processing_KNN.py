# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:22:32 2023

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
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=10, metric = "minkowski", p = 2)
classifier.fit(X_train, y_train)
# Prediccion de los resultados con el conjunto de Testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusion
from sklearn.metrics import confusion_matrix

#Diagonal nos dice las predicciones correctas, las otras nos dicen que son predicciones incorrectas 
cm = confusion_matrix(y_test, y_pred,labels=[0,1])#Orden de los labels determinanr la diagonal
TP = cm[0][0]
TN = cm[1][1]
FP = cm[0][1]
FN = cm[1][0]
Accuracy = (TP+TN)/(TP+TN+FP+FN)
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1_Score = 2*Precision*Recall/(Precision+Recall)
print(Accuracy,Precision,Recall,F1_Score)
