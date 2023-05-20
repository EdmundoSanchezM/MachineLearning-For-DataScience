# ACP

#importando dataset
dataset = read.csv('Wine.csv')

#Dividir data set entre conjunto de entranamiento y testing
library(caTools)
set.seed(123) #Seleccionando semilla
#sample.split(Datos a separar, porcentaje para entrenar)
split = sample.split(dataset$Customer_Segment,SplitRatio = 0.80)
#Separando datos de entrenamiento y testing
training_set = subset(dataset,split==TRUE)
testing_set = subset(dataset,split==FALSE)

#Escalado de valores
training_set[,1:13] = scale(training_set[,1:13])
testing_set[,1:13] = scale(testing_set[,1:13])

# Proyeccion de las componentes principales
#install.packages("caret")
library(caret)
library(e1071)
pca = preProcess(x = training_set[, -14], method = "pca", pcaComp = 2)
training_set = predict(pca,training_set)
training_set = training_set[, c(2,3,1)]
testing_set = predict(pca,testing_set)
testing_set = testing_set[, c(2,3,1)]

# Ajustar el modelo de regresión logística con el conjunto de entrenamiento.
classifier = svm(formula = Customer_Segment ~ . , 
                 data = training_set, 
                 type = "C-classification", 
                 kernel = "linear")

# Predicción de los resultados con el conjunto de testing
#Probabilidades de compra o no compra
y_pred = predict(classifier,newdata = testing_set[,-3])

# Crear la matriz de confusión
cm = table(testing_set[, 3], y_pred)

# Visualización del conjunto de entrenamiento
#install.packages("ElemStatLearn")
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.02)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.02)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'Clasificación (Conjunto de Entrenamiento)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', ifelse(y_grid == 2,'tomato','deepskyblue')))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', ifelse(set[, 3] == 2,'red3','blue3')))


# Visualización del conjunto de testing
set = testing_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.02)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.02)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'Clasificación (Conjunto de Entrenamiento)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', ifelse(y_grid == 2,'tomato','deepskyblue')))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', ifelse(set[, 3] == 2,'red3','blue3')))
