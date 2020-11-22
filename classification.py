import numpy as np 
from sklearn import datasets 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
iris = datasets.load_iris()
x = iris.data[:, [2,3]]
y = iris.target

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

from sklearn.linear_model import Perceptron 
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)
ppn.fit(x_train, y_train)
y_pred = ppn.predict(x_test)
print('error %s'%(y_test != y_pred).sum())
from sklearn.metrics import accuracy_score 
print('accuracy %s'%(accuracy_score(y_test, y_pred)))


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000, random_state=0)
lr.fit(x_train, y_train)
y_pred = lr.predict_proba(x_test[0,:].reshape(1, -1))
print('y prob', y_pred)

weights, params = [], []
for c in np.arange(-5, 5, dtype=np.float):
	lr = LogisticRegression(C=10**c, random_state=0)
	lr.fit(x_train, y_train)
	weights.append(lr.coef_[1])
	params.append(10**c)

weights = np.array(weights)
plt.figure(figsize=(6,4))
plt.plot(params, weights[:, 0], label='petal length')
plt.plot(params, weights[:, 1], '--', label='petal width')
plt.ylabel('weight coefficient', fontsize=12)
plt.xlabel('C', fontsize=12)
plt.legend(loc=0, fontsize=12)
plt.xscale('log')
plt.tight_layout()
plt.show()

from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)
print('y_pred', y_pred)

np.random.seed(0)
x_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(x_xor[:, 0]>0, x_xor[:, 1]>0)
y_xor = np.where(y_xor, 1, -1)

plt.figure(figsize=(6,4))
plt.scatter(x_xor[y_xor==1, 0], x_xor[y_xor==1, 1], c='b', marker='x', label='1')
plt.scatter(x_xor[y_xor==-1, 0], x_xor[y_xor==-1, 1], c='r', marker='s', label='-1')
plt.ylim(-3, 0)
plt.legend(loc=0)
plt.tight_layout()
plt.show()

svm = SVC(kernel='rbf', random_state=0, gamma=0.1, C=10.0)
svm.fit(x_xor, y_xor)


def gini(p):
	return p*(1-p)+(1-p)*(1-(1-p))

def entropy(p):
	return -p*np.log2(p)-(1-p)*np.log2(1-p)

def error(p):
	return 1-np.max([p, 1-p])

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = 'entropy', max_depth=3, random_state=0)
tree.fit(x_train, y_train)

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=-1)
forest.fit(x_train, y_train)


from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric = 'minkowski')
knn.fit(x_train, y_train)






