#-------------------------------------------------------------------------
# AUTHOR:           Andy Vu
# FILENAME:         clustering.py
# SPECIFICATION:    Run k-means multiple times and check which k value
#                   maximizes the Silhouette coefficient
# FOR:              CS 4210 - Assignment #5
# TIME SPENT:       05/04/2021-05/08/2021
#-------------------------------------------------------------------------

# importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) # reading the data by using Pandas library

# assign your training data to X_training feature matrix
X_training = df.copy()

best_k = 0
max_score = 0
s_score = []
best_kmeans = None

# run kmeans testing different k values from 2 until 20 clusters
# Use: kmeans = KMeans(n_clusters=k, random_state=0)
#      kmeans.fit(X_training)
for k in range(2, 21):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_training)
    # for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
    # find which k maximizes the silhouette_coefficient
    score = silhouette_score(X_training, kmeans.labels_)
    s_score.append(score)
    if score > max_score:
        best_k = k
        max_score = score
        best_kmeans = kmeans

# plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
k = range(2, 21)
plt.plot(k, s_score, "-o")
plt.show()

# reading the validation data (clusters) by using Pandas library
y_test = pd.read_csv("testing_data.csv", sep=",", header=None)

# assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
labels = np.array(y_test.values).reshape(1, len(y_test))[0]

# Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, best_kmeans.labels_).__str__())


# rung agglomerative clustering now by using the best value o k calculated before by kmeans
# Do it:
# agg = AgglomerativeClustering(n_clusters=<best k value>, linkage='ward')
# agg.fit(X_training)
agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
agg.fit(X_training)

# Calculate and print the Homogeneity of this agglomerative clustering
print("Agglomerative Clustering Homogeneity Score = " + metrics.homogeneity_score(labels, agg.labels_).__str__())
