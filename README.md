# Cluster them out!

### An Apache Spark (Scala) workflow for outlier detection, using K-means clustering.


##Overview

In this repository you can find Scala source code for automatic outlier detection on a specific dataset containing 5
clusters of various shapes but similar density. It is known from beforehand that the dataset contains several outliers.
In order to detect outliers, we perform K-means clustering on the data, for k >> 5 (the actual number of clusters).

The source code was originally developed by Lazaros Gogos and Vasilis Papastergios as a University assignment in
the Course of [Data Warehouses & Mining](https://qa.auth.gr/en/x/class/1/600237204). The authors attended the course 
during their 7th semester of B.Sc. studies at the 
[Aristotle University of Thessaloniki (AUTh)](https://www.csd.auth.gr/en/). 


## Basic Information

The solution presented in the current repository is adapted to the special characteristics of the given dataset and,
thus, it may not be applicable to other datasets that differ importantly from the given one. However, it is 
expected that the solution can be applied (with possible need of hyper-parameter tuning/selection) in similar datasets
with good performance. In particular, the solution is expected to perform properly on datasets that:
1. contain 3 - 6 actual clusters, ideally 5
1. (optionally) have clusters of different shape
1. have clusters with approximately the same density
1. contain a few outliers (relatively to the dataset total size)


## Implementation details

We use K-means clustering as the tool for detecting anomalies in the data. Knowing from beforehand the actual number of
clusters (5), gives us the opportunity to use a much larger hyper-parameter k (200), in a try to identify small, compound 
sub-clusters, overcoming possible problems imposed by the different cluster shapes.

<p align="center">
    <img align="middle" src="https://github.com/Bilpapster/cluster-them-out/blob/master/Images/clusters.png" alt="A scatter plot of two hundred clusters"/>
</p>

Based on the latter clusters, we compute the mean distance μ and standard deviation σ of all the data points from
the respective cluster center. We consider outliers to be all data points that have distance grater than μ + a * σ,
where a is a hyper-parameter in range [2, 4]. For the given dataset, we use a = 3.5. One can experiment with different
values of this hyper-parameter to achieve the desired performance on different but similar datasets.

<p align="center">
    <img align="middle" src="https://github.com/Bilpapster/cluster-them-out/blob/master/Images/outliers.png" alt="A scatter plot five clusters and several detected outliers"/>
</p>


## Future Improvements

1. In our approach, for every data point we compute the distance from its cluster center. An idea that could possibly
improve the robustness of our solution is to compute the distance to more points (e.g. 100 randomly selected points
of the respective cluster), rather than only one point (the cluster center). In that way, we can possibly 
avoid False Positive values that can arise when the cluster center is non-centrally (spatially) located in the 
sub-cluster.
   
2. Use pipeline automation tool (such as Apache Airflow) to integrate R scripts for applying further operations on the
data, such as removing outliers and/or finding the actual clusters.


## Further Information

The interested reader can find a presentation file in the root folder of the current repository. The authors were 
selected by the course Professor [Anastasios Gounaris](https://datalab-old.csd.auth.gr/~gounaris/) to hold an in-class
presentation of their work, among 20 teams.

