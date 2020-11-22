import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
path= '../results/'

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
iris = datasets.load_iris()
x =  iris.data[:,2:]
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(x, y)

# from sklearn.tree import export_graphviz 
# export_graphviz( tree_clf,
# 	out_file=image_path("iris_tree.dot"), 
# 	feature_names=iris.feature_names[2:],
# 	class_names=iris.target_names, 
# 	rounded=True,
#     filled=True
#     )

tree_clf.predict_proba([[5, 1.5]])
tree_clf.predict([[5, 1.5]])

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(x,y)