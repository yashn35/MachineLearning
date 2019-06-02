import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame({
    'x': [-1,-2,-3],
    'y': [39, 36, 30]
})

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, verbose=1)
kmeans.fit(df)

df1 = pd.DataFrame({
    'x': [-2],
    'y': [36]
})

labels = kmeans.predict(df1)
centroids = kmeans.cluster_centers_

fig = plt.figure(figsize=(5, 5))

colors = map(lambda x: colmap[x+1], labels)

plt.scatter(df['x'], df['y'], color=colors, alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()

print(labels)
