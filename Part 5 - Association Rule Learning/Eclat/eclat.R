# Eclat

# Preprocesado de Datos
dataset = read.csv("Market_Basket_Optimisation.csv", header = FALSE)

#Pasando a transaciones
library(arules)
dataset = read.transactions("Market_Basket_Optimisation.csv",
                            sep = ",", rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 100)

# Entrenar algoritmo Eclat con el dataset
rules = eclat(data = dataset, 
                parameter = list(support = 0.003, minlen = 2))

# Visualizacion de los resultados
inspect(sort(rules, by = 'support')[1:10])


