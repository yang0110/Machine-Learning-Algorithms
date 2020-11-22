import pandas as pd 

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 

x, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

import numpy as np 
cov_mat = np.cov(x_train.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('eigenvals', eigen_vals)

tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

import matplotlib.pyplot as plt 
plt.figure(figsize=(6,4))
plt.bar(range(1, 14), var_exp, alpha=0.5, align='center', label='ind explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid', label='cum explained variance')
plt.ylabel('Explained Variance')
plt.xlabel('Principle Components')
plt.legend(loc=0)
plt.show()

from sklearn.lda import LDA 
lda = LDA(n_components=2)
x_train_lda = lda.fit_transform(x_train, y_train)