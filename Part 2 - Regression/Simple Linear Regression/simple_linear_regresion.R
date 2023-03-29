# Regresion lineal simple

#importando dataset
dataset = read.csv('Salary_Data.csv')

#Dividir data set entre conjunto de entranamiento y testing
library(caTools)
set.seed(123) 
split = sample.split(dataset$Salary,SplitRatio = 2/3)
#Separando datos de entrenamiento y testing
training_set = subset(dataset,split==TRUE)
testing_set = subset(dataset,split==FALSE)

#Ajustar el modelo de regresion lineal simple con el conjunto de entrenamiento
#Variable dependiente en funcion(~) a la independiente
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)
summary(regressor)

#Predecir resultados con el conjunto de tests
#Funcion generica. Indicar modelo y data (data con el minos nombre de columnas)
y_pred = predict(regressor, newdata=testing_set)
y_pred

library(ggplot2)
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = "red") +#colocando puntos
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata=training_set)),
            colour = "blue") + #Colocando recta de regresion
  ggtitle("Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)")+
  xlab("Años de Experiencia") + 
  ylab("Sueldo (en $)")

#Visualizacion de los resultados en el conjunto de testing
ggplot() + 
  geom_point(aes(x = testing_set$YearsExperience, y = testing_set$Salary),
             colour = "red") +#colocando puntos
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata=training_set)),
            colour = "blue") + #Colocando recta de regresion
  ggtitle("Sueldo vs Años de Experiencia (Conjunto de Testing)")+
  xlab("Años de Experiencia") + 
  ylab("Sueldo (en $)")
