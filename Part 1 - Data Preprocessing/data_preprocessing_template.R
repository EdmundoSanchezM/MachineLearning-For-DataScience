# Plantilla Preprocesado de datos

#importando dataset

dataset = read.csv('Data.csv')
# dataset = dataset[,2:3]
#Dividir data set entre conjunto de entranamiento y testing
library(caTools)

set.seed(123) #Seleccionando semilla
#sample.split(Datos a separar, porcentaje para entrenar)
split = sample.split(dataset$Purchased,SplitRatio = 0.8)
#Separando datos de entrenamiento y testing
training_set = subset(dataset,split==TRUE)
testing_set = subset(dataset,split==FALSE)

#Escalado de valores
#[filas,columnas]
# training_set[,2:3] = scale(training_set[,2:3])#[Todas las filas, de 2 a 3]
# testing_set[,2:3] = scale(testing_set[,2:3])



