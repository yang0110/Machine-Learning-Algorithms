import pandas as pd 
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)

from sklearn.preprocessing import LabelEncoder
x = df.iloc[:, 2:].values 
y = df.iloc[:,1].values 
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe_lr = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=2)), ('clf', LogisticRegression(random_state=1))])
pipe_lr.fit(x_train, y_train)
print('accuracy', pipe_lr.score(x_test, y_test))

# import numpy as np 
# from sklearn.model_selection import StratifiedKFold
# kfold = StratifiedKFold(y=y_train, n_folds=10, random_state=1)
# scores = [] 

# for k, (train, test) in enumerate(kfold):
# 	pipe_lr.fit(x_train[train], y_train[train])
# 	score = pipe_lr.score(x_train[test], y_train[test])
# 	scores.append(score)
# 	print('fold %s, acc %s'%(k+1), score)
# print('CV Accur %s, %s'%(np.mean(scores), np.std(scores)))

from sklearn.cross_validation import cross_val_score
scores = cross_val_score(estimator=pipe_lr, x=x_train, y=y_train, cv=10, n_jobs=-1)

from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state=1))])
param_range = [0.001, 0.01, 0.1, 1.0, 10, 100, 1000]
