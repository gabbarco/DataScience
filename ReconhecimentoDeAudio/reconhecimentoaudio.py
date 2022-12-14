import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import soundfile as sf
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
file = ''  #Leitura do arquivo

path = os.getcwd()  #cria uma string com o diretório atual

#abrir o arquivo

data, sr= sf.read(file) #data é um Array

#calcula o periodo da amostragem

dt = 1/sr

#verificando os atributos do Array

data.ndim  #retorna o número de dimensões
data.shape #retorna o tamanho de cada dimensão
data.size #retorna o tamanho total do Array

#atribuindo os dados a um dataset NumPy

np.data = data

#plotar

#criar um vetor com tamanho igual ao da amostra

x = np.arange(681627)

#criar o vetor de dados

y = np.data[0:681627,0]

yMod = np.abs(y)


#visualizando o sinal

#criando um vetor de tempo
#inicio = 0 fim = tf, passo = dt

tf = 681627*dt #tempo final em segundos
t = np.arange(0,tf,dt)

## plotando o sinalno tempo

plt.plot(t,y) 
plt.show()

plt.plot(t,yMod)
plt.show()

## Aplicar a função smooth.spine() em todo o sinal não é adequado
##o melho é aplicar a função em partes menores do sinal

##criar janelas de tamanho 1000
d = 1000

#Zero padding

vetZero = np.zeros(1000)
 
vetSinalZeroP = np.concatenate((yMod,vetZero))

vetSinalZeroP.shape

n = 682627

s = np.arange(0,n,d)

S = np.zeros(2)

#fazer a primeira janela
i = 1
b = vetSinalZeroP[s[i-1]:(s[i])]
a = np.arange(s[i-1],(s[i]))
z = UnivariateSpline(a, b, s = 1e-5)
S=np.c_[a,z(a)]

s.shape
num = np.arange(2,683)

for i in num:
  b = vetSinalZeroP[s[i-1]:(s[i])]
  a = np.arange(s[i-1],(s[i]))
  z = UnivariateSpline(a, b, s = 1e-5)
  ss=np.c_[a,z(a)] #equivalente ao cbind
  S = np.concatenate([S,ss])
  
#fazer a última janela
b = vetSinalZeroP[s[i]:n]
a = np.arange(s[i],n)
z = UnivariateSpline(a, b, s = 1e-5)
ss=np.c_[a,z(a)]
S = np.concatenate([S,ss])

S.shape

tf = 682627*dt # tempo final em segundos
t = np.arange(0,tf,dt)

print("\nDetecção do não\n")

# Intervalo especíico
num = np.arange(0,682527)
for i in num:
  if t[i] == 3.81:
    idxInic = i
for i in num:  
  if t[i] == 4.18:
    idxFim = i

sinalPad = S[idxInic:idxFim,1]
    
plt.plot(t[idxInic:idxFim],sinalPad)

plt.show()

sinalPad.shape
m = 16317

S.shape

n = 682627

D = np.zeros(682627)

num = np.arange(0,n-m)

for i in num :
  sCorr = S[i:(i+m),1]
  v = np.corrcoef(sinalPad,sCorr)
  D[i] = v[0,1]

plt.plot(t,D)
plt.show()
num = 682527
i=88200
p=0
tp=[]
while (i<num): 
  if D[i] >= 0.42:
    print("Positivo!")
    tp.append(i*dt)
    print(tp[p])
    p=p+1
    i=i+16317 
  i=i+1

i=0
vp=0
fp=0
##Validação  
for i in range(5):
  if tp[i]>2 and tp[i]<3:
    vp=vp+1
  if tp[i]>3 and tp[i]<4.5:
    vp=vp+1
  if tp[i]>4.8 and tp[i]<5.5:
    fp=fp+1
  if tp[i]>6 and tp[i]<6.8:
    vp=vp+1
  if tp[i]>7 and tp[i]<8:
    fp=fp+1
  if tp[i]>8 and tp[i]<9:
    fp=fp+1
  if tp[i]>9 and tp[i]<10:
    fp=fp+1
  if tp[i]>10.5 and tp[i]<11.3:
    vp=vp+1
  if tp[i]>11.5 and tp[i]<12.3:
    vp=vp+1
  if tp[i]>12.5 and tp[i]<14:
    fp=fp+1

vn=vp
fn=fp

print("\n                    valor previsto\n                  negativo    positivo\n___________________________________________\n       negativo          {}            {}\n                 __________________________\nvalor real\n      positivo           {}            {}\n".format(vn,fp,fn,vp))

acc= (vp+vn)/vp+vn+fp+fn

print("\nA acurácia do valor medido é igual a ",acc)