# Plantilla Preprocesado de datos

#importando dataset

dataset = read.csv('Data.csv')

#eliminando NAs
#ave Subconjunto de X promediados a partir de FUN. 
#mean es promedio, donde na.rm = TRUE se usa para omitar los NA

dataset$Age = ifelse(is.na(dataset$Age),  # if the Age value is missing
       ave(dataset$Age, FUN = function(x) mean(x,na.rm=TRUE)),  # calculate the mean of Age, excluding missing values
       dataset$Age)  # else, use the original Age value

dataset$Salary = ifelse(is.na(dataset$Salary),  # if the Salary value is missing
       ave(dataset$Salary, FUN = function(x) mean(x,na.rm=TRUE)),  # calculate the mean of Salary, excluding missing values
       dataset$Salary)  # else, use the original Salary value


# Codificar datos categoricos
# Se especifican  los niveles y las etiquetas de los valores de la columna $X. 
# Los niveles son los valores únicos que aparecen en la columna $X 
# y se especifican en el argumento levels como un vector de caracteres 
# Las etiquetas son los valores enteros que se asignarán a cada nivel, 
# y se especifican en el argumento labels como un vector de enteros
dataset$Country = factor(dataset$Country,
                         levels = c("France","Spain","Germany"),
                         labels = c(1,2,3))
dataset$Purchased = factor(dataset$Purchased,
                         levels = c("No","Yes"),
                         labels = c(0,1))

#Dividir data set entre conjunto de entranamiento y testing
library(caTools)

set.seed(123) #Seleccionando semilla
#sample.split(Datos a separar, porcentaje para entrenar)
split = sample.split(dataset$Purchased,SplitRatio = 0.8)
#Separando datos de entrenamiento y testing
training_set = subset(dataset,split==TRUE)
testing_set = subset(dataset,split==FALSE)

#Escalado de valores
#[filas,columnas]
training_set[,2:3] = scale(training_set[,2:3])#[Todas las filas, de 2 a 3]
testing_set[,2:3] = scale(testing_set[,2:3])

















