import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier

# ler o arquivo 
pesquisa = pd.read_csv("", sep=";")  #Leitura do arquivo

df = pesquisa.drop(['age','balance','day','month','duration','campaign','pdays','previous'], axis=1)

treinamento = df[0:3021]

rotulo = treinamento.get('y')
teste = df[3021:]

treinamento= treinamento.drop(['y'],axis=1)
X_train = pd.get_dummies(treinamento,drop_first=True)

classificador = KNeighborsClassifier(n_neighbors = 8)#instanciar o modelo
classificador.fit(X_train,rotulo)#treinar o modelo

teste= teste.drop(['y'], axis=1)

X_test = pd.get_dummies(teste,drop_first=True)

from sklearn.metrics import confusion_matrix

y_true= teste = df.y[3021:]

y_pred= (classificador.predict(X_test))

mcNB = confusion_matrix(y_true, y_pred)

print(mcNB)

acur = (mcNB[0,0] + mcNB[1,1]) / len(y_true)

print("Acurácia knn: ",acur*100, "%")