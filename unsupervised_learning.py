from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.cluster.hierarchy import fcluster
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

""" Basic technique is clustering
Unlike supervised learning training data is not labeled.
Unsupervised learning is a type of machine learning that allows algorithms to discover hidden patterns and insights from unlabeled data without explicit guidance
 """
# Read the dataset
iris_df = pd.read_csv('Iris.csv')
iris_df.set_index('Id', inplace=True)
print (iris_df.head())

# Split data into Input variable and Target variable
X = iris_df.iloc[:, :-1].reset_index(drop=True)
print ("Input variables:")
print (X)
y = iris_df.iloc[:,-1].reset_index(drop=True)
print ("Target variable:")
print (y)
print ("List of target variable:")
species = list(y)
print (species)

# Create a model
model = KMeans(n_clusters=3)
model.fit(X)
labels = model.predict(X)
print ("List of labels:")
print (labels)

print ("Cluster center")
print (model.cluster_centers_)

# Mean of each cluster is called centroid
# Scatter plot between PetalLengthCm vs SepalLengthCm.
xs = iris_df.iloc[:,2]
print (xs)
print (type(xs))
ys = iris_df.iloc[:,0]
print (ys)
print (type(ys))
plt.scatter(xs, ys, c=labels)
plt.show()

species_label_df = pd.DataFrame({'Species': species, 'Labels': labels})
print (species_label_df)


ct = pd.crosstab(species_label_df['Labels'], species_label_df['Species'])
print (ct)

""" Inertia is the way to measure how tightly the clusters are bound togather. It measures how far the samples are from the centriod.
Lower value of inertia is always better. """
print ("Inertia of the model {}".format(model.inertia_))

# Try to find the inertia of the cluster for different  n_clusters values
inertia_list = []
cluster_count_list = []
for i in range (1, 15):
    model = KMeans(n_clusters=i)
    model.fit(X)
    labels = model.predict(X)
    cluster_count_list.append(i)
    inertia_list.append(model.inertia_)
print (inertia_list)

plt.plot(cluster_count_list, inertia_list, linestyle='--', marker='o',)
plt.show()


# Read the dataset
column_names = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280', 'Proline']
wine_df = pd.read_csv('wine/wine.csv', names=column_names, header=None)
#iris_df.set_index('Id', inplace=True)
print (wine_df)
model = KMeans(n_clusters=3)
labels = model.fit_predict(wine_df)
print (labels)

# wine_list = ['barbera', 'barolo', 'grignolino']
# wine_label_df = pd.DataFrame({'wines': wine_list, 'Labels': labels})
# ct = pd.crosstab(wine_label_df['Labels'], wine_label_df['wines'])
# print (ct)

variance_all = wine_df.var()
print ("Variance of all data")
print (variance_all)

scalar = StandardScaler()
scalar.fit(wine_df)
StandardScaler(copy=True, with_mean=True, with_std=True)
wine_scaled = scalar.transform(wine_df)
print (wine_scaled)
""" 
Steps:
===============
1. StandardScaler
2. KMeans """

scalar = StandardScaler()
kmeans = KMeans(n_clusters=3)
pipelines = make_pipeline(
    scalar,
    kmeans
)
pipelines.fit(wine_scaled)
labels = pipelines.predict(wine_scaled)
print (labels)

""" 
Hierarchical Clustering
==========================
Hierarchical clustering is a technique used to group items based on their similarity. This process involves the 
formation of clusters in a step-by-step manner.

Agglomerative Clustering is one of them
One common approach to hierarchical clustering is agglomerative clustering. In this method, each item starts as its 
own cluster. The algorithm then repeatedly merges the nearest clusters together.

Initial Clustering
In the initial configuration, each country is treated as a separate cluster. As the agglomerative process unfolds, 
clusters are combined according to their similarity until a stopping criterion is reached. Finally, it is a big cluster.
 """

# import matplotlib.pyplot as plt
# from scipy.cluster.hierarchy import linkage, dendrogram
# merging = linkage(samples, method='complete')
# dendrogram(merging,
#            labels=country_name,
#             leaf_rotation=90,
#              leaf_font_size=6 )
# plt.show()

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import linkage, dendrogram

###################################################################
# Generate sample data with three distinct, spherical clusters.   #
###################################################################
X, y = make_blobs(
    n_samples=200,      # Total number of data points
    n_features=10,       # Number of features (x and y coordinates)
    centers=10,          # The number of clusters to generate
    cluster_std=1.0,    # The standard deviation of the clusters
    random_state=42     # For reproducibility
)
# X is the variable that holds the raw data that you want to cluster.
print (X)
# y is the array of true cluster labels for the generated data.
print(y)

""" Perform hierarchical clustering (Agglomerative)
The 'linkage' function calculates the "distance" between clusters at each step of the hierarchical clustering process.
The dendrogram visualizes the hierarchical clustering process, showing the sequence of merges and the distances at 
which they occurred. """
# When the method is used as ward
merging = linkage(X, method='ward')
dendrogram(merging,
           truncate_mode='lastp',
           show_leaf_counts=True,
           leaf_rotation=90,
           leaf_font_size=6 )
plt.show()


labels = fcluster(merging, 15, criterion='distance')
print ("\nLabels retrieved from the clusters:\n {}".format(labels))

# When the method is used as complete
merging = linkage(X, method='complete')
dendrogram(merging,
           truncate_mode='lastp',
           show_leaf_counts=True,
           leaf_rotation=90,
           leaf_font_size=6 )
plt.show()

labels = fcluster(merging, 15, criterion='distance')
print ("\nLabels retrieved from the clusters:\n {}".format(labels))

""" t-NSE - t-distribution stochastic neighbor embeeding
It maps data from high dimensional space to 2 or 3dimensional space for easier visualization.
it approximately preserves the nearness of samples. """
print ("Input variables:")
print (X)
model = TSNE(learning_rate=10)
transformed = model.fit_transform(X)
print (transformed)
plt.scatter(transformed[:,0], transformed[:,1])
plt.show()

print ("# Understand how kmean works - If we are feeding the model data series")
sample = np.array([[1, 2, 3, 0, 3, 2, 1],
                  [1, 0, 0, 2, 3, 1, 3],
                  [0, 2, 2, 3, 2, 3, 0],
                  [3, 1, 3, 0, 1, 2, 0]
                  ])
print ("\nSample data:\n{}".format(sample))
kmeans = KMeans(n_clusters=3)
kmeans.fit(sample)
print("\nLabels found:\n{}".format(kmeans.labels_))

print ("# Understand how kmean works - If we are feeding the model dataframe")
sample_df = pd.DataFrame({"Length" :[1, 2, 3, 0, 3, 2, 1],
                  "Height": [1, 0, 0, 2, 3, 1, 3],
                  "Width": [0, 2, 2, 3, 2, 3, 0],
                  "Weight": [3, 1, 3, 0, 1, 2, 0]
                  })
print (sample_df)
X = sample_df[['Length', 'Height', 'Width', 'Weight']]
kmeans = KMeans(n_clusters=3)
kmeans.fit(sample)
print("\nLabels found:\n{}".format(kmeans.labels_))
