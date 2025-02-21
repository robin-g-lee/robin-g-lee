from sklearn.preprocessing import StandardScaler
import numpy as np
data = pd.read_csv('Week_2_Data/Iris.csv')
data = data[['SepalLengthCm','PetalLengthCm','Species']]
species = data['Species']
features = data.iloc[:, :-1]
features = features.values.astype("float32", copy = False)

stscaler = StandardScaler().fit(features)
features = stscaler.transform(features)

# Elbow plot:
k=2 # number of columns
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=k+1).fit(features) 
distances, indices = nbrs.kneighbors(features)
distances = np.sort(distances[:, k])

plt.plot(distances)
plt.xlabel("Points sorted by distance")
plt.ylabel("k-NN distance")
plt.title("k-NN Distance Plot")
plt.show()
# min_samples is 12, because first jump (knee) is between 10 and 15
# eps is 0.5

dbsc = DBSCAN(eps = .5, min_samples = 12).fit(features)
labels = dbsc.labels_
core_samples = np.zeros_like(labels, dtype = bool)
core_samples[dbsc.core_sample_indices_] = True
data['label'] = labels
print(data['label'].unique())
print(data.head())

plt.figure(figsize=(8, 6))
# Plot the clusters
colors = ['indianred','#57db5f','#5f57db']
for label in data['label'].unique():
    cluster_data = data[data['label'] == label]
    plt.scatter(cluster_data['SepalLengthCm'], cluster_data['PetalLengthCm'], 
                label=f'Class {label}',
                color = colors[label % len(colors)],
               s = 150)
plt.legend()
plt.title("Iris Data")

