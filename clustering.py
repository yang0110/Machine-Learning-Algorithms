from sklearn.datasets import make_blobs 

x, y = make_blobs(n_samples=150, n_features=2, cluster_std=0.5, centers=3, shuffle=True, random_state=0)

import matplotlib.pyplot as plt
plt.figure() 
plt.scatter(x[:,0], x[:,1], c='white', marker='o')
plt.show()

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, max_iter=300)
y_km = km.fit_predict(x)

distortions = []
for i in range(1, 11):
	km = KMeans(n_clusters=i, init='k-means++')
	km.fit(x)
	distortions.append(km.inertia_)

plt.figure()
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('distortion')
plt.show()