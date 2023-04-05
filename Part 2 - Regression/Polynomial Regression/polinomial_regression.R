# Regresion Polinomica

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
#[filas,columnas]
#training_set[,2:3] = scale(training_set[,2:3])#[Todas las filas, de 2 a 3]
#testing_set[,2:3] = scale(testing_set[,2:3])

#Ajustar el modelo de regresion lineal simple con el conjunto de datos 
lin_reg = lm(formula = Salary ~ .,
               data = dataset)
summary(lin_reg)

#Ajustar el modelo de regresion polinomica con el conjunto de datos
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
dataset$Level5 = dataset$Level^5
poly_reg = lm(formula = Salary ~ .,
              data = dataset)
summary(poly_reg)

#Visualizacion del modelo lineal
library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = "red") +#colocando puntos
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata=dataset)),
            colour = "blue") + #Colocando recta de regresion
  ggtitle("Prediccion lineal del sueldo en funcion del nivel del empleado")+
  xlab("Nivel del empleado") + 
  ylab("Sueldo (en $)")

#Visualizacion del modelo polinomico
library(ggplot2)
x_grid = seq(min(dataset$Level),max(dataset$Level),0.1)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = "red") +#colocando puntos
  geom_line(aes(x = x_grid, y = predict(poly_reg, newdata=data.frame(Level = x_grid,
                                                                     Level2 = x_grid^2,
                                                                     Level3 = x_grid^3,
                                                                     Level4 = x_grid^4,
                                                                     Level5 = x_grid^5))),
            colour = "blue") + #Colocando recta de regresion
  ggtitle("Prediccion polinomica del sueldo en funcion del nivel del empleado")+
  xlab("Nivel del empleado") +
  ylab("Sueldo (en $)")


#Prediccion de nuevos resultado con Regrsion Lineal
y_pred = predict(lin_reg,newdata=data.frame(Level = 6.5))

#Prediccion de nuevos resultado con Regrsion Polinomica
y_pred = predict(poly_reg,newdata=data.frame(Level = 6.5, Level2 = 6.5^2, 
                                             Level3 =6.5^3, Level4 = 6.5^4,
                                             Level5 = 6.5^5))

