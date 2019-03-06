# libraries
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt

# read csv file
iris = pd.read_csv('data01_iris.csv')

###########################################################
# one-sample t-test
###########################################################

d = np.array([-0.31,-0.67,-0.61,-2.07,-1.31])
r = sp.stats.ttest_1samp(d,0)
r.pvalue

# data01_iris.csv 에서 setosa 품종의 평균 Sepal Length가 4가 아니라는 
# 가설에 대한 p-value를 구하시오.

d = iris [iris['Species']=='setosa']
d.shape
v = d['Sepal.Length']
v.mean()
d = np.array(v)
r = sp.stats.ttest_1samp(v, 4)
r


###########################################################
# two-sample t-test
###########################################################

np.random.seed(1)
x1 = np.random.normal(0.5,1,10)    # mean, standard error, count
x2 = np.random.normal(-0.5,1,10)
r = sp.stats.ttest_ind(x1,x2)  # t-test individual
r.pvalue
# what is the number of samples for pvalue < 0.01

np.random.seed(1)
x1 = np.random.normal(0.5,1,20)    # mean, standard error, count
x2 = np.random.normal(-0.5,1,20)
r = sp.stats.ttest_ind(x1,x2)  # t-test individual

np.random.seed(1)
x1 = np.random.normal(0,1,100)    # mean, standard error, count
x2 = np.random.normal(0,1,100)
r = sp.stats.ttest_ind(x1,x2)  # t-test individual


iris1 = iris[ iris['Species']=='versicolor' ]
iris2 = iris[ iris['Species']=='setosa' ]
sp.stats.ttest_ind(iris1['Sepal.Length'],iris2['Sepal.Length'])

# what is the p-value between Petal Length of versicolor and virginica 
# for samples of which Petal Length > 4

petlen_versi = iris[ 
        (iris['Species'] == 'versicolor') & (iris['Petal.Length'] > 4.0)
                    ]

petlen_virgi = iris[ 
        (iris['Species'] == 'virginica') & (iris['Petal.Length'] > 4.0) 
                    ]

sp.stats.ttest_ind(petlen_versi['Petal.Length'], petlen_virgi['Petal.Length'])


###########################################################
# correlation test
###########################################################

np.random.seed(2)
z = np.random.normal(0,1,10)
x = 1*z + np.random.normal(0,1,10)
y = 1*z + np.random.normal(0,1,10)
plt.plot(x, y, 'bo')

r = sp.stats.pearsonr(x,y)    # Standard Deviation by Pearson
r[0] # correlation coefficient
r[1] # p-value of correlation test
# what is the number of samples that makes p-value < 0.001

# Increase Count
# Or, increase k of k*z + np.random.normal(0, 1, 10)

sp.stats.pearsonr(iris['Sepal.Width'],iris['Petal.Length'])

# what is the correlation and p-value between Sepal Length and Petal Width of setosa?

seto = iris[ (iris['Species']=='setosa')]
sp.stats.pearsonr(seto['Sepal.Length'], seto['Petal.Width'])
plt.plot(seto['Sepal.Length'], seto['Petal.Width'], 'bo')
        
###########################################################
# chisq test
###########################################################

np.random.seed(3)
w = 1
z = np.random.normal(0,1,10)
x = w*z + np.random.normal(0,1,10)
y = w*z + np.random.normal(0,1,10)
x2 = pd.cut(x,3)
y2 = pd.cut(y,3)
tbl = pd.crosstab(x2,y2)
r = sp.stats.chisquare(tbl,axis=None)
r.pvalue
# what is the weight for p-value < 0.001?

x = pd.cut(iris['Sepal.Width'],3)
x = pd.cut(iris['Sepal.Width'],3,labels=['Short', 'Medium', 'Long'])
y = iris['Species']
tbl = pd.crosstab(x,y)
sp.stats.chisquare(tbl,axis=None)

# Sepal.Width와 Petal.Length 사의 correlation test의 pvalue
sp.stats.pearsonr(iris['Sepal.Width'],iris['Petal.Length'])

# Sepal.Width와 Petal.Length를 pd.cut을 이용하여 level 4개짜리 
# 범주형 데이터로 변형한 후 chi-square test를 하셔서 
# pvalue를 구하시오. 


###########################################################
# one-way anova (f-test)
###########################################################

iris1 = iris[iris['Species']=='setosa']
iris2 = iris[iris['Species']=='versicolor']
iris3 = iris[iris['Species']=='virginica']

sp.stats.f_oneway(iris1['Sepal.Length'],iris2['Sepal.Length'],iris3['Sepal.Length'])

# 카테고리가 두개라면 ttest냐 ftest냐?
sp.stats.f_oneway(iris1['Sepal.Length'],iris2['Sepal.Length'])
sp.stats.ttest_ind(iris1['Sepal.Length'],iris2['Sepal.Length'])




###########################################################
# Practice
###########################################################

# Practice 1
s = 0
for i in range(1,5000):
    p = xxx
    if p<0.01: s = s+1



# Practice 2

data = pd.read_csv('data04_carseat.csv')
sales = data['Sales']
comp_price = data['CompPrice']
income = data['Income']
ad = data['Advertising']
pop = data['Population']
price = data['Price']
age = data['Age']
edu = data['Education']

sales_bin = data['SalesBin']
shel_loc = data['ShelveLoc']
urban = data['Urban']
us = data['US']

sp.stats.pearsonr(sales, comp_price)
sp.stats.pearsonr(sales, income)
sp.stats.pearsonr(sales, ad)
sp.stats.pearsonr(sales, pop)
sp.stats.pearsonr(sales, price)
sp.stats.pearsonr(sales, age)
sp.stats.pearsonr(sales, edu)

shloc_good = data[data['ShelveLoc']=='Good']
shloc_mid = data[data['ShelveLoc']=='Medium']
shloc_bad = data[data['ShelveLoc']=='Bad']
sp.stats.f_oneway(shloc_good['Sales'], shloc_mid['Sales'], shloc_bad['Sales'])

urb_y = data[data['Urban']=='Yes']
urb_n = data[data['Urban']=='No']
sp.stats.ttest_ind(urb_y['Sales'], urb_n['Sales'])
sp.stats.f_oneway(urb_y['Sales'], urb_n['Sales'])

us_y = data[data['US']=='Yes']
us_n = data[data['US']=='No']
sp.stats.ttest_ind(us_y['Sales'], us_n['Sales'])




sbhigh = data[data['SalesBin']=='High']
sblow = data[data['SalesBin']=='Low']

sp.stats.ttest_ind(sbhigh['CompPrice'], sblow['CompPrice'])
sp.stats.ttest_ind(sbhigh['Income'], sblow['Income'])
sp.stats.ttest_ind(sbhigh['Advertising'], sblow['Advertising'])
sp.stats.ttest_ind(sbhigh['Population'], sblow['Population'])
sp.stats.ttest_ind(sbhigh['Price'], sblow['Price'])
sp.stats.ttest_ind(sbhigh['Age'], sblow['Age'])
sp.stats.ttest_ind(sbhigh['Education'], sblow['Education'])

tbl = pd.crosstab(sales_bin, shel_loc)
sp.stats.chisquare(tbl, axis=None)

tbl = pd.crosstab(sales_bin, urban)
sp.stats.chisquare(tbl, axis=None)

tbl = pd.crosstab(sales_bin, us)
sp.stats.chisquare(tbl, axis=None)












# PLEASE DO NOT GO DOWN BEFORE YOU TRY BY YOURSELF

###########################################################
# Practice Reference Code
###########################################################

import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt


# Practice 1
df = pd.read_csv('data03_breastcancer.csv')
d1 = df[ df['stage']==0 ]
d2 = df[ df['stage']==1 ]
sp.stats.ttest_ind(d1['DCT'],d2['DCT'])


s = 0
for i in range(1,5000):
    v1 = d1.iloc[:,i]
    v2 = d2.iloc[:,i]
    r = sp.stats.ttest_ind(v1,v2)
    p = r.pvalue
    if p<0.01: s = s+1



# practice 2
df = pd.read_csv('data04_carseat.csv')

# case 2
plist = []
y = df['SalesBin']
for i in range(2,df.shape[1]):
    v = df.iloc[:,i]
    if type(v[0]) == type('string'): # categorical
        tbl = pd.crosstab(y,v)
        r = sp.stats.chisquare(tbl,axis=None)
    else:
        v1 = v[ y=='High' ]
        v2 = v[ y=='Low' ]
        r = sp.stats.ttest_ind(v1,v2)
    plist.append(r.pvalue)

RES = pd.DataFrame({"Name":df.columns[2:df.shape[1]], "Pvalue":plist})





















