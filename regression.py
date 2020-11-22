import pandas as pd 

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE', 'DIS', 'RAD',
'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set(stype='white')

cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size=2.5)
plt.show()

import numpy as np 
cm = np.corrcoef(df[cols].values.T)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=cols, xticklabels=cols)
plt.show()


x = df['RM'].values 
y = df['MEDV'].values
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
sc = StandardScaler()
x = StandardScaler.fit_transform(x)
lr.fit(x, y)

from sklearn.linear_model import RANSACRegressor 
ransac = RANSACRegressor(LinearRegression(), max_trials=1000, min_samples=50)
ransac.fit(x, y)

from sklearn.linear_model import Ridge 
ridge = Ridge(alpha=0.1)
from sklearn.linear_model import Lasso 
lasso = Lasso(alpha=1.0)
from sklearn.linear_model import ElasticNet 
en = ElasticNet(alpha=1.0, l1_ratio=0.5)

from sklearn.tree import DecisionTreeRegressor 
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(x, y)



