import numpy as np
import random  
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

data = pd.read_csv('iris-with-errors.csv', header=(0))
print(data)

data= data.replace('?', np.nan)

X=np.array(data[data.columns[0:data.shape[1]-1]],dtype=float)

medias = np.nanmean(X, axis=0)

for i in np.arange(0,X.shape[0]):
    for j in np.arange(0,X.shape[1]):
        if (np.isnan(X[i,j])== True):
            X[i,j] = medias[j]

data.info()
print(data.describe())

data= datasets.load_iris()
data= pd.DataFrame(data=np.c_[data['data'], data['target']], columns=data['feature_names']+['target'])

x= np.array(data[data.columns[0:data.shape[1]-1]], dtype=float)

scaler = MinMaxScaler(feature_range=(0,1))
x_pad= scaler.fit_transform(x)
print(x_pad)

df = pd.DataFrame({'A':['a','b','c','a','b','c']})
print(df.head())

df= pd.get_dummies(df)
print(df.head())

                