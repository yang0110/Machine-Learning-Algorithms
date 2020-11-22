from sklearn import datasets
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, LabelEncoder 
iris = datasets.load_iris()
x, y = iris.data[50:, [1,2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import cross_val_score 

import numpy as np 
clf1 = LogisticRegression(penalty='l2', C=0.001,
	random_state=0)

clf2 = DecisionTreeClassifier(max_depth=1,
	criterion='entropy',
	random_state=0)

clf3 = KNeighborsClassifier(n_neighbors=1,
	p=2,
	metric='minkowski')

pipe1 = Pipeline([['sc', StandardScaler()],
		['clf', clf1]])

pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])

clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']

for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
	scores = cross_val_score(estimator=clf, X=x_train,
	y=y_train,
	cv=10, scoring='roc_auc')
	print(scores.mean(), scores.std())


# mv_clf = MajorityVoteClassifier( classifiers=[pipe1, clf2, pipe3])

# clf_labels += ['Majority Voting']

# all_clf = [pipe1, clf2, pipe3, mv_clf]


# for clf, label in zip(all_clf, clf_labels):
# 	scores = cross_val_score(estimator=clf, X=X_train,
# 	y=y_train,
# 	cv=10, scoring='roc_auc')
# 	print(scores.mean(), scores.std())

import pandas as pd

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol',
					'Malic acid', 'Ash',
					'Alcalinity of ash',
					'Magnesium', 'Total phenols',
					'Flavanoids', 'Nonflavanoid phenols',
					'Proanthocyanins',
					'Color intensity', 'Hue',
					'OD280/OD315 of diluted wines',
					'Proline']
df_wine = df_wine[df_wine['Class label'] != 1]
y = df_wine['Class label'].values
x = df_wine[['Alcohol', 'Hue']].values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)


from sklearn.ensemble import BaggingClassifier
tree = DecisionTreeClassifier(criterion='entropy',
		max_depth=None)
bag = BaggingClassifier(base_estimator=tree,
						n_estimators=500,
						max_samples=1.0,
						max_features=1.0,
						bootstrap=True,
						bootstrap_features=False,
						n_jobs=1,
						random_state=1)
from sklearn.metrics import accuracy_score
tree = tree.fit(x_train, y_train)
y_train_pred = tree.predict(x_train)
y_test_pred = tree.predict(x_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)

print(tree_train, tree_test)

bag = bag.fit(x_train, y_train)
y_train_pred = bag.predict(x_train)
y_test_pred = bag.predict(x_test)
bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)

print(bag_train, bag_test)

from sklearn.ensemble import AdaBoostClassifier
tree = DecisionTreeClassifier(criterion='entropy',
		max_depth=None)

ada = AdaBoostClassifier(base_estimator=tree, n_estimators=100)

tree = tree.fit(x_train, y_train)
y_train_pred = tree.predict(x_train)
y_test_pred = tree.predict(x_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print(tree_train, tree_test)


ada = ada.fit(x_train, y_train)

y_train_pred = ada.predict(x_train)
y_test_pred = ada.predict(x_test)
ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
print(ada_train, ada_test)



