# Apriori

# Preprocesado de Datos
dataset = read.csv("Market_Basket_Optimisation.csv", header = FALSE)
#Craando matriz dispersa. A que fila, A que columnaa, A que valor
#install.packages("arules")
library(arules)
dataset = read.transactions("Market_Basket_Optimisation.csv",
                            sep=",",
                            rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset,topN=100)

#Entrenar algoritmo Apriori con el dataset
#s: Al menos 3 ventas diarias por semana / total de ventas - Se puede hablar
#c: Se maneja por defecto
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

#Visualizacion de los resultados
inspect(sort(rules, by = 'lift',decreasing = TRUE)[1:10])

#install.packages("arulesViz")
library(arulesViz)
plot(rules, method = "graph", engine = "htmlwidget")
