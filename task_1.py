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
        print(self.df)

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
        print(df)
        self.X = np.array(df)
        self.Y = np.array(Y)
        X = self.X
        random.seed(123)
        df = df.reset_index()
        X_scale = sklearn.preprocessing.scale(X)
        
        print('here')
        df = df.reset_index()
        #hop = pyclustertend.hopkins(X,len(X))
        print('here')
        
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
        plt.title('Clusters of Customers')
        plt.xlabel('Annual Income(k$)')
        plt.ylabel('Spending Score(1-100')
        plt.show()
        return 0    
    

driver = task_1('./minute_weather.csv')
driver.kMeansModel()