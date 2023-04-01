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
SL = 0.05
regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
                data=dataset)
summary(regression)

regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
                data=dataset)
summary(regression)

regression = lm(formula = Profit ~ R.D.Spend + Marketing.Spend, data=dataset)
summary(regression)

regression = lm(formula = Profit ~ R.D.Spend , data=dataset)
summary(regression)


