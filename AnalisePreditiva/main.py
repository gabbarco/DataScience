import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier

# ler o arquivo 
pesquisa = pd.read_csv("", sep=",", decimal=".")  #Leitura do arquivo
print(pesquisa)
df = pesquisa.drop(['name'], axis=1)

treinamento1 = df[0:24]
print(treinamento1)
treinamento2= df[30:len(df)]
print(treinamento2)
treinamento= pd.concat([treinamento1,treinamento2])
print(treinamento)
rotulo = treinamento.get('status')
print(rotulo)
teste = df[24:30]
print(teste)

treinamento= treinamento.drop(['status'], axis=1)

classificador = KNeighborsClassifier(n_neighbors = 33)#instanciar o modelo

classificador.fit(treinamento,rotulo)#treinar o modelo

print("Status do paciente isolado")
teste= teste.drop(['status'], axis=1)
print(classificador.predict(teste))#predição dos dados teste