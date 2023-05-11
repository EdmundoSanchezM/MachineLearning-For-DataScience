# Natural Language Processing

# Importar el data set
dataset_original = read.delim("Restaurant_Reviews.tsv", quote = '',
                              stringsAsFactors = FALSE)

# Limpieza de textos
#install.packages("tm")
#install.packages("SnowballC")
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))
# Consultar el primer elemento del corpus
# as.character(corpus[[1]])
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords(kind = "en"))
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

# Crear el modelo Bag of Words
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)

dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

# Codificar la variable de clasificación como factor
dataset$Liked = factor(dataset$Liked, levels = c(0,1))

# Ajustar el clasificador con el conjunto de entrenamiento.
library(e1071)
classifier = svm(formula = Liked ~ .,
                 data = training_set,
                 type = "C-classification",
                 kernel = "linear",
                 scale = FALSE)

# Predicción de los resultados con el conjunto de testing
y_pred = predict(classifier, newdata = testing_set[,-692])

# Crear la matriz de confusión
cm = table(testing_set[, 692], y_pred)

TP <- cm[1,1]
TN <- cm[2,2]
FP <- cm[1,2]
FN <- cm[2,1]
Accuracy <- (TP+TN)/(TP+TN+FP+FN)
Precision <- TP/(TP+FP)
Recall <- TP/(TP+FN)
F1_Score <- 2*Precision*Recall/(Precision+Recall)
print(cm)
cat(paste("Accuracy:", Accuracy, "Precision:", Precision, "Recall:", Recall, "F1-Score:", F1_Score, sep=" "))
