# image-clustering-in-fashion-mnist-dataset-
Apply a number of different clustering algorithms on raw and dimensionally reduced data of fashion images and evaluate their performance regarding specific metrics.


### Problem description
We want to apply a number of different clustering algorithms on raw and dimensionally reduced data of fashion images and evaluate their performance regarding specific metrics.

### Dataset description
link: https://keras.io/api/datasets/fashion_mnist/
Fashion-MNIST is a dataset of fashion images consisting of a training set of 60,000 examples and a test set of 10,000 examples (Fig. 1) Each example is a 28x28 grayscale image, associated with a label from 10 classes.


### Dimensionality reduction techniques
After scaling all measured features using the normalisation technique (values [0-1]), we applied the following dimensionality techniques to the dataset:

Principal component analysis (PCA):  Principal component analysis is a statistical process that allows reducing the dimensionality of a dataset while preserving as much of the original variability as possible. This is achieved by transforming the data into a new coordinate system, where the axes are the principal components. However, interpreting the principal components may not always be straightforward, especially when dealing with a large number of features.

Factor analysis (FA): While PCA aims to capture the maximum variance in the data, Factor Analysis focuses on identifying the underlying factors that contribute to the observed variables. The factor analysis describes the covariance associations among many variables in terms of a few underlying, but unobservable, random quantities called factors. Factor analysis assumes that the variables can be grouped by their correlations.

Stacked autoencoder (SAE): It is a nonlinear generalisation of PCA that uses an adaptive, multilayer encoder network to transform the high-dimensional data into a low-dimensional code and a similar decoder network to recover the data from the code.

Convolutional stacked autoencoder (CSAE): A Convolutional Stacked Autoencoder (CSAE) is an extension of the traditional stacked autoencoder architecture, specifically designed for handling structured grid-like data such as images. The advantage of using convolutional layers is that they enable the autoencoder to capture local patterns and spatial dependencies, which is crucial for image data.

Fast algorithm for Independent Component Analysis (Fast ICA):  ICA aim is similar in many aspects to principal component analysis (PCA) and factor analysis. The key difference of ICA from PCA and FA is that ICA aims for the statistical independence of the resulting components, while the components obtained using PCA and FA are only linearly independent, which in general does not imply statistical independence. A computationally very efficient method performing the actual estimation of ICA is given by the FastICA algorithm (ref.)


### Classification methods used in this work
Minibatch K-Means: The Mini-batch K-means partitioning-based algorithm is a variant of the traditional K-means clustering algorithm that is designed to work with large datasets more efficiently. While the standard K-means algorithm processes the entire dataset in each iteration, mini-batch K-means processes only a random subset or "mini-batch" of the data in each iteration.

DBSCAN: Density-Based Spatial Clustering of Applications with Noise is a density-based clustering algorithm for grouping together data points that are close to each other in high-dimensional space. Unlike K-means, DBSCAN does not require specifying the number of clusters beforehand and can find clusters of arbitrary shapes.

Agglomerative Hierarchical Clustering: It is a type of clustering algorithm that builds a hierarchy of clusters. It is a bottom-up approach, meaning that it starts with individual data points and progressively merges them into larger clusters. The result is a tree-like structure known as a dendrogram.

Expectation-Maximization algorithm (EM) for Gaussian Mixture Model (EM-GMM): It is a model-based approach for clustering when the data is assumed to be generated from a mixture of several Gaussian distributions (K-Means is a special case of GMM). In order to find the parameters of the Gaussian for each cluster (e.g the mean and standard deviation), Expectation–Maximization (EM) optimization algorithm is used. 

Bisecting K-Means: Bisecting K-Means is a hybrid approach between partitional and hierarchical top down clustering. Instead of partitioning the data set into K clusters in each iteration, it splits one cluster into two sub clusters at each bisecting step (by using k-means) until k clusters are obtained.


### Performance Metrics
To validate the results of clustering methods, the following four indexes were used:

Calinski–Harabasz index: It provides a way to assess the compactness of clusters and the separation between clusters in a dataset. The index is calculated based on the ratio of the between-cluster variance to the within-cluster variance. The interpretation of the index is such that higher values indicate better-defined clusters.

Davies–Bouldin index: Similar to the Calinski–Harabasz index, the Davies–Bouldin index evaluates the compactness and separation between clusters in a dataset. The range of the Davies–Bouldin Index is theoretically between 0 and positive infinity. A lower DBI indicates more compact and well-separated clusters.

Silhouette score: The Silhouette Score is a metric used to calculate the goodness of a clustering technique, indicating how well-separated the clusters are. The Silhouette Score ranges from -1 to 1, where: A high Silhouette Score indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. A low Silhouette Score indicates that the object is poorly matched to its own cluster and well matched to neighboring clusters. A score around 0 indicates overlapping clusters.

Adjusted Rand Index (ARI): The Adjusted Rand Index (ARI) is a measure of the similarity between two data clusterings, adjusting for chance. The higher the ARI value, the closer the two clusterings are to each other. Even though ARI takes negative values, generally considered range for ARI is [0, 1]. ARI value 0 indicates real and modeled clustering do not agree on pairing, ARI value 1 indicates real and modeled clustering both represent the same clusters.


CHECK REPORT FOR THE RESULTS
