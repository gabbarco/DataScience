# abrir o arquivo
import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv("", sep=";", decimal=".") #Leitura do arquivo

#Implementação do algoritmo Kmeans

kmeans = KMeans(n_clusters=2, random_state=0).fit(df)

print(kmeans.labels_)