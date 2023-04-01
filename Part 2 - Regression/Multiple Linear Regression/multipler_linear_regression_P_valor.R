# Regresion Lineal Multiple 

#importando dataset
dataset = read.csv('50_Startups.csv')


# Codificar datos categoricos
dataset$State = factor(dataset$State,
                       levels = c("New York","California","Florida"),
                       labels = c(1,2,3))

#Dividir data set entre conjunto de entranamiento y testing
library(caTools)
set.seed(123) 
split = sample.split(dataset$Profit,SplitRatio = 0.8)
training_set = subset(dataset,split==TRUE)
testing_set = subset(dataset,split==FALSE)

#Ajustar el modelo de Regresion Lineal Multiple con el Conjunto de Entranmiento
#. todas las otras
regression = lm(formula = Profit ~ ., data=training_set)
#P valor = Pr(>|t|)
summary(regression)

# Predecir los resultados con el conjunto de testing
y_pred = predict(regression, newdata = testing_set)

# Construir un modelo optimo con la Eliminacion hacia atrasq
backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
backwardElimination(training_set, SL)


