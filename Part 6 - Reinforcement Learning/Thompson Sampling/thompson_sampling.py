# -*- coding: utf-8 -*-
"""
Created on Thu May  4 21:47:16 2023

@author: josue
"""

# Muestreo Thompson

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Algoritmo del Muestreo Thompson
d = 10 #Numero de categorias
N = 10000 #Numero de rondas/usuarios
number_of_rewards_1 = [0] * d
number_of_rewards_0 = [0] * d
ads_selected = []
total_reward = 0

for n in range(0,N):
    max_random = 0
    ad = 0
    for i in range(0,d):
        random_beta = random.betavariate(number_of_rewards_1[i]+1,number_of_rewards_0[i]+1)#Ojo aqui
        if(random_beta > max_random):
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    #En la vida real no es asi
    reward = dataset.values[n, ad]
    if(reward == 0):
        number_of_rewards_0[ad] = number_of_rewards_0[ad] + 1
    else:
        number_of_rewards_1[ad] = number_of_rewards_1[ad] + 1
    total_reward = total_reward + reward
# Visualizar los resultados - Histograma
plt.hist(ads_selected)
plt.title('Histograma de anuncios')
plt.xlabel('ID del Anuncio')
plt.ylabel('Frecuencia de visualizacion del anuncio')
plt.show()
