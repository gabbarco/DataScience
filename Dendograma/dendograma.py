from matplotlib import pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist

pesquisa = pd.read_csv("C:/Users/46456334810/DataScience/Dendograma/processed.cleveland.data", sep=",",decimal=".")
df= pesquisa.drop(["num","ca","thal"],axis=1)

d = pdist(df,metric='euclidean')

from scipy.cluster.hierarchy import linkage

my_cluster = linkage(d,'ward')

dn = dendrogram(my_cluster)

plt.show()
