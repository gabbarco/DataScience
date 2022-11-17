# abrir o arquivo
import pandas as pd
from sklearn.cluster import DBSCAN

df = pd.read_csv("", sep=";", decimal=".") #Leitura do arquivo

#Implementação do algoritmo DBscan

clustering = DBSCAN(eps= 5, min_samples= 2).fit(df)

print(clustering.labels_)