import sklearn.datasets as datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sb
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import sklearn.preprocessing
import scipy.cluster.hierarchy as sch
import pylab
import sklearn.mixture as mixture
import pyclustertend 
import random
import numpy as np
from reader import Reader
import random

class task_1(object):
    def __init__(self, csvFilePath):
        # Universal Doc
        self.csvDoc = csvFilePath
        # Classes
        R = Reader(csvFilePath)
        self.df = R.data

    def csvColumns(self):
        return self.df.dtypes

    def hopkins(self):
        hop = 1
        df = self.df.copy()
        df = df.drop(columns=[
            'hpwren_timestamp', 'min_wind_direction', 'max_wind_direction',
            'relative_humidity', 'min_wind_speed', 'max_wind_speed',
            'rain_duration', 'air_pressure'
            ])
        df = df.dropna()
        Y = df.pop('rowID')
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        df = df[indices_to_keep].astype(np.float64)
        self.X = np.array(df)
        self.Y = np.array(Y)
        X = self.X
        random.seed(123)
        df = df.reset_index()
        X_scale = sklearn.preprocessing.scale(X)
        df = df.reset_index()
        #hop = pyclustertend.hopkins(X,len(X))
        
        return X_scale
    
    def clusterNum(self):
        X_scale = self.hopkins()
        clustersNum = range(1,8)
        wcss = []
        for i in clustersNum:
            kmeans = cluster.KMeans(n_clusters=i)
            kmeans.fit(X_scale)
            wcss.append(kmeans.inertia_)

        plt.plot(clustersNum, wcss)
        plt.xlabel("Clusters")
        plt.ylabel("Score")
        plt.title("Clusters Amount")
        plt.show()

    def kMeansModel(self):
        X_scale = self.hopkins()
        kmeans = cluster.KMeans(n_clusters=5)
        y_kmeans = kmeans.fit_predict(self.X)
        X = self.X
        plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label ='Cluster 1')
        plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', label ='Cluster 2')
        plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', label ='Cluster 3')
        plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='cyan', label ='Cluster 4')
        plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=100, c='magenta', label ='Cluster 5')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label = 'Centroids')
        plt.title('Clusters of Weather')
        plt.show()
        return 0    
    
class task_2(object):
    def __init__(self, csvFilePath):
        # Universal Doc
        self.csvDoc = csvFilePath
        # Classes
        T1 = task_1(csvFilePath)
        self.hop = T1.hopkins()
        self.X = T1.X
    
    def silhouette_method_HC(self):
        X_scale = self.hop
        X = self.X
        range_n_clusters = [2, 3, 4]

        for n_clusters in range_n_clusters:
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            ax1.set_xlim([-0.1, 1])
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])


            clusterer = cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
            cluster_labels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            silhouette_avg = metrics.silhouette_score(X, cluster_labels)
            print(
                "For n_clusters =",
                n_clusters,
                "The average silhouette_score is :",
                silhouette_avg,
            )

            # Compute the silhouette scores for each sample
            sample_silhouette_values = metrics.silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(
                X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
            )

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(
                "Silhouette analysis for Hierarchical clustering on sample data with n_clusters = %d"
                % n_clusters,
                fontsize=14,
                fontweight="bold",
            )

        plt.show()  

    def mix_gaussians(self):
        X_scale = self.hop
        X = self.X

        gmm = mixture.GaussianMixture(n_components = 5).fit(X)
        labels = gmm.predict(X)

        plt.title("Grouping by Mixture of Gaussians")
        plt.scatter(X[:, 0], X[:, 1], c=labels,cmap="plasma")
        plt.show()


path = './minute_weather.csv'
driver = task_1(path)
driver_2 = task_2(path)
#driver.kMeansModel()
driver_2.mix_gaussians()
