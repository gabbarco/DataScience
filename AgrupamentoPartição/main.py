# abrir o arquivo
import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv("C:/Users/46456334810/DataScience-1/AgrupamentoPartição/mamografia.csv", sep=";", decimal=".")

#Implementação do algoritmo Kmeans

kmeans = KMeans(n_clusters=2, random_state=0).fit(df)

print(kmeans.labels_)