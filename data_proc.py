import pandas as pd 
from io import StringIO

csv_data = '''A, B, C, D
1.0, 2.0, 3.0, 4.0
5.0, 6.0,, 8.0 
0.0, 11.0, 12.0,'''
df = pd.read_csv(StringIO(csv_data))
print(df)

print(df.isnull().sum())
print(df.values)

print(df.dropna())
print(df.dropna(axis=1))
print(df.dropna(how='all'))
print(df.dropna(thresh=4))

# from sklearn.impute import SimpleImputer 
# imr = SimpleImputer(missing_values='NaN', strategy='mean')
# imr = imr.fit(df)

# imputed_data = imr.transform(df.values)
# print(imputed_data)

# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(categorical_features = [0])
# ohe.fit_transform(x).toarray()

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

df_wine.columns = ['Class label', 'Alcohol',
	'Malic acid', 'Ash',
	'Alcalinity of ash', 'Magnesium',
	'Total phenols', 'Flavanoids',
	'Nonflavanoid phenols',
	'Proanthocyanins',
	'Color intensity', 'Hue',
	'OD280/OD315 of diluted wines',
	'Proline']
print(df_wine.head())

from sklearn.model_selection import train_test_split

x = df_wine.iloc[:, 1:].values 
y = df_wine.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

from sklearn.preprocessing import MinMaxScaler 
mms = MinMaxScaler()
x_train = mms.fit_transform(x_train)
x_test = mms.transform(x_test)

from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit(x_test)


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l2', C=0.1)
lr.fit(x_train, y_train)
# y_pred = lr.predict(x_test)
# print('accuracy', lr.score(x_test, y_test))

from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
forest.fit(x_train, y_train)
importances = forest.feature_importances_
print('imp', importances)



