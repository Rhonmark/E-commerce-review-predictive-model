from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, centers= 4, random_state=42)

kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
 
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
plt.title('K-Means Clustering')
plt.show()