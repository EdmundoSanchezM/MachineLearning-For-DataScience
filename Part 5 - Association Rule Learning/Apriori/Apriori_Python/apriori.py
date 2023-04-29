# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 15:24:47 2023

@author: josue
"""

#Apriori

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Market_Basket_Optimisation.csv',header=None)
transactions = []

for i in range(0,len(dataset)):
    temporal = []
    line = dataset.iloc[i,:].values
    for j in range(0,len(line)):
        if(str(line[j]) == "nan"):
            break
        else:
            temporal.append(line[j])
    transactions.append(temporal)
    
# Entrenar el algoritmo de Apriori
from apyori import apriori
rules = apriori(transactions,min_support = 0.003, min_confidence = 0.2, 
                min_lift = 3 , min_length = 2)

#Visualizacion de los resultados
results = list(rules)
results.sort(key=lambda x:x.ordered_statistics[0].lift, reverse=True)

results[2]

rule = list()
support = list()
confidence = list()
lift = list()
 
for item in results:
    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    lenItems = len(items)
    if(lenItems == 1):
        rule.append(items)
    else:
        buildStr = ""
        for i in range(0,lenItems): 
            buildStr = buildStr + items[i]
            j = i+1
            if(j != lenItems ):
                buildStr =buildStr + " -> "
        rule.append(buildStr)
    
    #second index of the inner list
    support.append(str(item[1]))
 
    #third index of the list located at 0th
    #of the third index of the inner list
 
    confidence.append(item[2][0][2])
    lift.append(item[2][0][3])
 
output_ds  = pd.DataFrame({'rule': rule,
                           'support': support,
                           'confidence': confidence,
                           'lift': lift
                          }).sort_values(by = 'lift', ascending = False)
print(output_ds.iloc[[0,1],:].values)

