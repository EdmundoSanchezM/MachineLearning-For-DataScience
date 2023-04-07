# SVR


#importando dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[,2:3]

#Dividir data set entre conjunto de entranamiento y testing
#library(caTools)
#set.seed(123) #Seleccionando semilla
#sample.split(Datos a separar, porcentaje para entrenar)
#split = sample.split(dataset$Purchased,SplitRatio = 0.8)
#Separando datos de entrenamiento y testing
#training_set = subset(dataset,split==TRUE)
#testing_set = subset(dataset,split==FALSE)

#Escalado de valores
#training_set[,2:3] = scale(training_set[,2:3])#[Todas las filas, de 2 a 3]
#testing_set[,2:3] = scale(testing_set[,2:3])

#Ajustar el modelo de SVR con el conjunto de datos
#install.packages("e1071")
library(e1071)
regression = svm(formula = Salary ~ ., data = dataset, type = "eps-regression",
                 kernel = "radial")


#Prediccion de nuevos resultado con SVR
y_pred = predict(regression, newdata=data.frame(Level = 6.5))



#Visualizacion del modelo de SVR
library(ggplot2)
x_grid = seq(min(dataset$Level),max(dataset$Level),0.1)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = "red") +#colocando puntos
  geom_line(aes(x = x_grid, y = predict(regression, newdata=data.frame(Level = x_grid))),
            colour = "blue") + #Colocando recta de regresion
  ggtitle("Prediccion (SVR)")+
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")

