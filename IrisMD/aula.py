import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

data= pd.read_csv('iris-with-errors.csv', header=(0))
print("Linhas x colunas:",data.shape)
data.head(15)
