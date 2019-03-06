# libraries
import numpy as np
import matplotlib.pyplot as plt

# 균등분포
np.random.seed(1)
u = np.random.rand(100)
u.mean()
u.var()
plt.hist(u)

# 정규분포
np.random.seed(1)
z = np.random.randn(100)
z.mean()
z.var()
plt.hist(z)

z = np.random.randn(100)*2 + 1

# 이변량
np.random.seed(2)
x = np.random.randn(50)
y = np.random.randn(50)
plt.plot(x,y,'bo')
np.cov(x,y)
np.corrcoef(x,y)

z = np.random.randn(50)
x = z + np.random.randn(50)
y = z + np.random.randn(50)
plt.plot(x,y,'bo')
np.cov(x,y)
np.corrcoef(x,y)

x1 = 2*x
y1 = 3*y
np.cov(x1,y1)
np.corrcoef(x1,y1)

# 연습문제
# 두 변수 사이의 상관관계 계수가 -0.8보다 작도록 x와 y를 생성하시오. 

###########################################################
# Practice
###########################################################






















# PLEASE DO NOT GO DOWN BEFORE YOU TRY BY YOURSELF

###########################################################
# Practice Reference Code
###########################################################


# Practice
import numpy as np
import pandas as pd

# read data
df = pd.read_csv('data01_iris.csv')

#



