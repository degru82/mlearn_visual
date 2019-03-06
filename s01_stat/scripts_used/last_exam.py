# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt

# 1 
data = pd.read_csv('data01_iris.csv')

vers = data[data['Species']=='versicolor']
vers_ptln = vers['Petal.Length']
vers_ptln.mean()

# 2
virg = data[data['Species']=='virginica']
virg_spwd = virg['Sepal.Width']

vers_spwd = vers['Sepal.Width']

r = sp.stats.ttest_ind(virg_spwd, vers_spwd)
r.pvalue


# 3
seto = data[data['Species']=='setosa']
seto_spln = seto['Sepal.Length']
seto_ptwd = seto['Petal.Width']

r = sp.stats.pearsonr(seto_spln, seto_ptwd)
r[1]

# 4
data = pd.read_csv('data10_solar.csv')

solar = data['Solar']

temp = data['Temp']
chill = data['Chill']
hindex = data['Hindex']
humid = data['Humid']
dewpt = data['Dewpt']
wind = data['Wind']
hiWind = data['HiWind']
rain = data['Rain']
barom = data['Barom']

corr_dict = {}
corr_dict['Temp'] = sp.stats.pearsonr(solar, temp)
corr_dict['Chill'] = sp.stats.pearsonr(solar, chill)
corr_dict['Hindex'] = sp.stats.pearsonr(solar, hindex)
corr_dict['Humid'] = sp.stats.pearsonr(solar, humid)
corr_dict['Dewpt'] = sp.stats.pearsonr(solar, dewpt)
corr_dict['Wind'] = sp.stats.pearsonr(solar, wind)
corr_dict['HiWind'] = sp.stats.pearsonr(solar, hiWind)
corr_dict['Rain'] = sp.stats.pearsonr(solar, rain)
corr_dict['Barom'] = sp.stats.pearsonr(solar, barom)

corr_sorted = sorted(corr_dict.items(), key=lambda x: x[1])
corr_sorted


# 5

from sklearn.linear_model import LinearRegression

X = data[['Temp', 'Chill']]
y = data['Solar']

lm = LinearRegression()
lm.fit(X,y)
lm.coef_



# 6

X = data.iloc[:,:-1]
y = data['Solar']

lm = LinearRegression()
lm.fit(X,y)
r2 = lm.score(X,y)
r2












r = lm.score()
