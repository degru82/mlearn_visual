# libraries
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

# read data
df = pd.read_csv('data05_boston.csv')

# simple linear regression
from sklearn.linear_model import LinearRegression
X = df[ ['lstat','age','rm'] ]
#X = df[ ['lstat'] ]   # Single row DataFrame
y = df['medv']    # Series

lm = LinearRegression()
lm.fit(X,y)    # Get beta-0 and beta-1: medv = b0 + b1*lstat
lm.coef_  # coefficients
lm.intercept_ # intercepter
# Training Session is Done

lm.predict(10)
lm.predict([[5], [10], [15]])    # 3 by 1 2dim arr (matrix), not 1 by 3 [5,10,15]
yhat = lm.predict(X) # prediction by using training set
r2 = lm.score(X,y) # R2; Performance Measurement named R-Square
rmse = np.sqrt(((y-yhat)**2).mean())    # error from training set
# r2, rmse are TRAINING ERROR, so if you consider more inputs,
# PERFORMANCE should be better


plt.plot(X,y,'bo')
plt.plot(X,yhat,'r',linewidth=2)
plt.title('%s vs. Medv: %.2f' % ('lstat',r2))
plt.show()

# multiple linear regression
X = df.iloc[:,0:13]
y = df['medv']
lm = LinearRegression()
lm.fit(X,y)
lm.coef_  # coefficients
lm.intercept_ # intercepter
yhat = lm.predict(X) # prediction
r2 = lm.score(X,y) # R2
rmse = np.sqrt(((y-yhat)**2).mean())

plt.plot(yhat,y,'bo')
plt.title('All vs. Medv: %.2f' % r2)
plt.show()

# Where is P value??????? what the f**k!!!
# So we are introducing StatsModel

# using StatsModel
import statsmodels.api as sm
X = df.iloc[:,0:13]
# While scikit-learn takes intercept in the default model,
#  statsmodels doesn't take it by default.
# If you consider intercept, you should invoke 'add_constant()'
X = sm.add_constant(X)
y = df['medv']

# Ordinary Least Squre
# This is a sort of constructor
f = sm.OLS(y,X)
r = f.fit()
r.summary()

r.params
r.pvalues

# add a new variable
plt.plot(df['lstat'],df['medv'],'bo')
X = df[ ['lstat'] ]
lstat2 = X['lstat']**2
X['lstat2'] = lstat2
X = sm.add_constant(X)
y = df['medv']
f = sm.OLS(y,X)
r = f.fit()
r.summary()

X = df[ ['lstat','rm'] ]
X['lstat_rm'] = X['lstat'] * X['rm']
X = sm.add_constant(X)
y = df['medv']
f = sm.OLS(y,X)
r = f.fit()
r.summary()


# training vs. test set
np.random.seed(1)
train_idx = list(np.random.choice(np.arange(df.shape[0]),300,replace=False))
test_idx = list(set(np.arange(df.shape[0])).difference(train_idx))
dftrain = df.iloc[train_idx,:]
dftest = df.iloc[test_idx,:]

# Removes inputs whose p-values are high
# This describes HOW SIMPLER MODEL CAN REDUCE OVERFITTING ISSUE.
xtrain = dftrain.iloc[:,[0,1,3,4,5,7,8,9,10,11,12]]
ytrain = dftrain['medv']
xtest = dftest.iloc[:,[0,1,3,4,5,7,8,9,10,11,12]]
ytest = dftest['medv']

# Alternatively,
# Use all samples in both TRAINING and TEST
# This causes the increase of TEST-SET RMSE
xtrain = dftrain.iloc[:,:-1]
ytrain = dftrain['medv']
xtest = dftest.iloc[:,:-1]
ytest = dftest['medv']


lm = LinearRegression()
lm.fit(xtrain,ytrain)

yhat_train = lm.predict(xtrain)
rmse_train = np.sqrt( ((ytrain-yhat_train)**2).mean() )
r2_train = lm.score(xtrain,ytrain)

yhat_test = lm.predict(xtest)
rmse_test = np.sqrt( ((ytest-yhat_test)**2).mean() )
r2_test = lm.score(xtest,ytest)

print(rmse_train,rmse_test)


###########################################################
# Practice Reference Code
###########################################################

# data01_iris.csv를 읽으시오. 
# Sepal Width ~ Sepal.Length + Petal.Length + Petal.Width 로 
# 선형 회귀 분석을 수행하시오. 
# (1) R2와 RMSE 값은 얼마인가?
# (2) 어떤 변수의 제곱항을 추가하였을 때, 가장 높은 R2를 갖는 것은 어느 변수인가?
# (3) Sepal.Length와 Petal.Length의 interaction 항을 추가하였을 때, R2은 
# 얼마인가?
# (4) 범주형 변수 Sepcies를 포함시켜 선형 회귀 분석을 수행하시오.
x = np.zeros(10)
x[0:5] = 1

df = pd.read_csv('data01_iris.csv')

X = df [['Sepal.Length', 'Petal.Length', 'Petal.Width']]
y = df ['Sepal.Width']

lm = LinearRegression()
lm.fit(X, y)
r2 = lm.score(X,y) # R2; Performance Measurement named R-Square

yhat = lm.predict(X)
rmse = np.sqrt(((y-yhat)**2).mean())    # error from training set


r2_1storder = r2

spln_2nd = df['Sepal.Length']**2
ptln_2nd = df['Petal.Length']**2
ptwd_2nd = df['Petal.Width']**2

X = df [['Sepal.Length', 'Petal.Length', 'Petal.Width']]
X['spln_2nd'] = spln_2nd
lm = LinearRegression()
lm.fit(X,y)
r2 = lm.score(X,y)

r2_spln2 = r2

X = df [['Sepal.Length', 'Petal.Length', 'Petal.Width']]
X['ptln_2nd'] = ptln_2nd
lm = LinearRegression()
lm.fit(X,y)
r2 = lm.score(X,y)

r2_ptln2 = r2

X = df [['Sepal.Length', 'Petal.Length', 'Petal.Width']]
X['ptwd_2nd'] = ptwd_2nd
lm = LinearRegression()
lm.fit(X,y)
r2 = lm.score(X,y)

r2_ptwd2 = r2

print(r2_1storder, r2_spln2, r2_ptln2, r2_ptwd2)



splnptln = X['Sepal.Length'] * X['Petal.Length']

X = df [['Sepal.Length', 'Petal.Length', 'Petal.Width']]
X['splnptln'] = splnptln
lm = LinearRegression()
lm.fit(X,y)
r2 = lm.score(X,y)

r2_inter = r2

print(r2_1storder, r2_spln2, r2_ptln2, r2_ptwd2, r2_inter)


r2_org = r2

seteff_idx = (df['Species'] == 'setosa')
vrgeff_idx = (df['Species'] == 'virginica')

X = df [['Sepal.Length', 'Petal.Length', 'Petal.Width']]
y = df ['Sepal.Width']

seteff_idx = seteff_idx.astype(int)
vrgeff_idx = vrgeff_idx.astype(int)

X['seteff'] = seteff_idx
X['vrgeff'] = vrgeff_idx

lm = LinearRegression()
lm.fit(X,y)
r2 = lm.score(X,y)



X = sm.add_constant(X)
f = sm.OLS(y,X)
r = f.fit()
r.summary()





















# PLEASE DO NOT GO DOWN BEFORE YOU TRY BY YOURSELF

###########################################################
# Practice Reference Code
###########################################################

# practice
df = pd.read_csv('data01_iris.csv')

# OLS
X = df[['Sepal.Length','Petal.Length','Petal.Width']]
y = df['Sepal.Width']

lm = LinearRegression()
lm.fit(X,y)
lm.score(X,y)

X = sm.add_constant(X)
f = sm.OLS(y,X)
f.fit().summary()


# adding 2nd order term
X = df[['Sepal.Length','Petal.Length','Petal.Width']]
y = df['Sepal.Width']
x1 = X['Petal.Width']**2
X['PW2'] = x1

X = sm.add_constant(X)
f = sm.OLS(y,X)
f.fit().summary()


# adding interaction term
X = df[['Sepal.Length','Petal.Length','Petal.Width']]
y = df['Sepal.Width']

x1 = X['Petal.Width']*X['Petal.Length']
X['Inter'] = x1

X = sm.add_constant(X)
f = sm.OLS(y,X)
r = f.fit()
r.summary()


# adding Species
X = df[['Sepal.Length','Petal.Length','Petal.Width']]

x1 = pd.Series(np.zeros(X.shape[0]))
x1[ df['Species']=='setosa' ] = 1

x2 = pd.Series(np.zeros(X.shape[0]))
x2[ df['Species']=='virginica' ] = 1

X['Species_setosa'] = x1
X['Species_virginica'] = x2
y = df['Sepal.Width']

lm = LinearRegression()
lm.fit(X,y)
r2 = lm.score(X,y)

X = sm.add_constant(X)
f = sm.OLS(y,X)
r = f.fit()
r.summary()








