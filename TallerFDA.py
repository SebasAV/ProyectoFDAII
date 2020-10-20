# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def train_test_split(indep, dep, test_size, random_state):
    indep_train = []
    dep_train = []
    indep_test = []
    dep_test = []
    
    indep_test = indep.sample(frac=test_size)
    
    indep_test = indep.loc[indep.index.isin(indep_test.index)]
    dep_test = dep.loc[dep.index.isin(indep_test.index)]
    indep_train = indep.loc[~indep.index.isin(indep_test.index)]
    dep_train = dep.loc[~dep.index.isin(dep_test.index)]
    
    indep_test.sort_index(ascending=True)
    dep_test.sort_index(ascending=True)
    indep_train.sort_index(ascending=True)
    dep_train.sort_index(ascending=True)
    
    return indep_train, indep_test, dep_train, dep_test

data = pd.read_csv("datos_clima.csv", index_col=0, parse_dates=True, dayfirst=True)
data.sort_index(ascending=True)
data.index.name = "Fecha"
print(f"Tabla Data:\n\n {data.head()} \n")

# Se revisa que no existan valores NaN o NA
print(f"La cantidad de datos nulos es: {data.isnull().sum().sum()}. \n")
print(f"La cantidad de NA es: {data.isna().sum().sum()} \n")

# Se revisan los tipos de datos de cada columna
print("Resumen de variables:\n\n", data.dtypes ,"\n")

# Se saca el promedio por cada 7 datos y se filtra extrayendo los valores por cada 7 dias
dataRolled = data.rolling(7).mean()
data2 = dataRolled.iloc[6::7, :]
print(data2.head())

# Se grafica la matriz de autocorrelacion
corr = data2.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Se separa la variable dependiente de las independientes
dep_var = data2.loc[:,"Temperatura (C)"]
indep_vars = data2.loc[:, data2.columns != "Temperatura (C)"]

train_x, test_x, train_y, test_y = train_test_split(indep_vars, dep_var, test_size=0.3, random_state=1)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)