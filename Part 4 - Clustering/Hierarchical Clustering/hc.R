# Clusterting Jerárquico

#importando dataset

dataset = read.csv('Mall_Customers.csv')
X = dataset[,4:5]

#Utilizar el dendrograma para encotrar el numero optimo de clusters
dendrogram = hclust(d = dist(X, method = "euclidean"),
                   method = "ward.D")
plot(dendrogram,
     main = "Dendrograma", 
     xlab = "Clientes del centro comercial", 
     ylab = "Distancia Euclida")

# Ajustar el clustering jerarquico a nuestro dataset
hc = hclust(d = dist(X, method = "euclidean"),
                   method = "ward.D")
y_hc = cutree(tree = hc, k = 5)
#Visualizacion de los clusters
#install.packages("cluster")
library(cluster)
clusplot(X,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 4,
         plotchar = FALSE,
         span = TRUE,
         main = "Clustering de clientes",
         xlab = "Ingresos anuales",
         ylab = "Puntuacion (1-100)")
