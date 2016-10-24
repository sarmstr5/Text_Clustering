import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn import cluster
from sklearn import preprocessing
from sklearn import pairwise
from heapq import nsmallest
from sys import stdout
import random
from datetime import datetime as dt
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool, cpu_count
import itertools
from threading import Thread


class shanes_kmeans:
    """
    K-means is a methodology of minimizing encoding/decoding error of X
    ui is mean point of cluster i
    Znk = z is the binary indicator variable that is 1 if object n == cluster k, 0 otherwise
    sum(Znk) is therefore 1
    ui = sum(Znk * Xi) / sum(Znk)
    Xi is assigned to cluster k that gives the min distance
    e.g. (Xn - uk)T(Xn - uk)
    """

    cluster_centers = []
    labels = []
    SSE = 0

    # note -  * tuple w/ positional args, ** dictionary/keyword args
    def __init__(self, k_clusters=3, tolerance=1e-4, random_state=4, max_iter=10, n_cores = cpu_count(), *args, **kwargs):
        """
        :param data: df of data to be clustered
        :param k_clusters: int number of clusters
        :param tolerance: float distance until convergence
        :param random_state: int seed of random generator
        :return: numpy array of clustered classes
        """
        self.obj_d = {'k_clusters': k_clusters, 'tolerance': tolerance, 'random_state': random_state,
                      'max_iter': max_iter, 'n_cores': n_cores}

        if kwargs is not None:
            for key in kwargs:
                self.obj_d[key] = kwargs[key]
            if key == 'grid':
                print("in grid part of kwargs")

    def fit(self, data, correct_classifier=None):
        '''
        :param data: df of data to be clustered
        :param correct_classifier: numpy array of correct classifications
        :return numpy array of clustered classes
        '''
        empty_v = np.ones(len(data))
        df = data.insert(loc=len(data.columns), column='cluster', value=empty_v) #inserts 0s at end of df
        self.obj_d['data'] = data
        self.obj_d['correct_classifier'] = correct_classifier
        max_js = []
        min_js = []
        for j in data:
            min_j = data.loc[:,j].min()
            min_js.append(min_j)
            max_j = data.loc[:,j].max()
            max_js.append(max_j)
        self.obj_d['min_js'] = min_js
        self.obj_d['max_js'] = max_js

    def compute_cluster_mean(x_k):
        pass

    def initial_clusters(data):
        k_clusters = self.obj_d['k_clusters']
        have_a_median_cluster = random.choice([True, False])
        min_js = self.obj_d['min_js'] # min column values from data
        max_js = self.obj_d['max_js']
        initial_cluster_locs = [[0]*data.columns.max() for k in range(k_clusters)] # k by j 0 matrix
        n_rows = len(data) #assuming df
        #grab random points to be initial locations
        if(have_a_median_cluster):
            for j in data.columns:
                initial_cluster_locs[k][j] = (min_js[j] + max_js[j])/2  # may not be necessary calc if normalized
            random_xs = random.sample(range(nrows), k-1) #random index list w/out replacement
            initial_cluster_locs = data[random_xs]  #locations are random from data points

        else:
            random_xs = random.sample(range(nrows), k) #random index list w/out replacement
            initial_cluster_locs = data[random_xs]  #locations are random from data points

        return initial_cluster_locs
        #can also try using hierarchical clustering first or choose points farthest away

    def find_min_k(cluster_means, xi):
        pass

    def partition_slices(end, n_partitions):
        slices = [slice(i, i + n_partitions) for i in range(0, end, n_partitions)]
        return slices

    def cluster_rows(X_df, slice_i, cluster_means):
        X = X_df.iloc[slice_i, :]
        # cluster_means = self.obj_d['cluster_means']
        # calc distance
        for row in X_df:
            find_closest_cluster(row, cluster_means)

        # find min_k

    def find_closest_cluster(row, cluster_means):
        distances = []
        for cluster in cluster_means:
            distance = find_distance(cluster, row)

    def find_cluster_proximity(cluster_locations, row_entity, similarity=True):
        proximity = 0
        k_num = 0
        cluster = []
        if(similarity):
            # proximity measured with cosine, goal is to maximize the sum of the cosine similarity of an
            # object to its cluster centroid
            # can use a weighted average for the position
            i = 0
            for k in cluster_locations:
                #use cdist!!!!!!
                prox = pairwise.cosine_similarity(row, cluster) #returns a scaler?
                if prox > proximity:
                    proximity = prox
                    cluster = k
                    row.cluster=i
                i += 1

        else:
            # use different proximity metric such as bregman divergence
            # proximity = 1/d
            pass
        return proximity

    def kmeans(X, k_clusters, max_runs = 50, verbose=False, tolerance=1e-2):
        """
        Predict cluster for xi
        :param X: data to be predicted
        :param k_clusters: numnber of clusters
        :param max_runs: maximum number of runs to find convergence
        :param verbose: for print statements
        :param tolerance: acceptable difference between centriods for convergence
        :return numpy array labels
        """

        #create centriods
        # initialize random cluster means
        # set number of random seeds for running kmeans multiple times
        # run single_run_means in parrellel
        # create pool and use seeds as array to parrallel
        # for numner of random seeds
        # call kmeans and get SSE centers and number of iteranions
        # join pool
        # get track best run with metric SSB, silhouette coefficient, calinski-harabaz index

    def kmeans_run(self, X):
        """
        Predict cluster for xi
        :param X: data to be predicted
        :return numpy array labels
        """
        converged = False
        best_metric, best_x, best_centers = 0, 0, 0 #return best score
        data = self.obj_d['data']
        centriods = initial_clusters(data)

        # distance to closest center
        distances_to_cluster = np.zeros(X.shape[0])
        x_labels = find_distances(x, centriods, proximity_type)
        # create list of closest neigbhors to center


        # assign remaining centriods to points with likely probability from each other
        # compute distancves to centriods
        # subtract mean from values
        for i in maxiterations:
            find SSB
            find mean of centers
            if SSB is less than min SSB
                save SSB, x, and centers
            check if means moved
                if not break
        return converged SSB, x, and centers


        #3 Update cluster means uk
        #4 Start over
        # slices = partition_slices(len(data), cpu_count())   #list indices for multiprocessing

        # while(not converged):
        #     for row in data.values.tolist():
        #         #1 for xi find k that minimizes D, closest cluster mean, set Znk =1, Znj = 0
        #         pool = Pool()
        #         # adds closest cluster
        #         # what is this finding for me the best cluster....
        #
        #         # change to threading
        #         results = pool.map(find_min_k, (cluster_locs, row))
        #         pool.close()
        #         pool.join()

       #2 If all of assignments Znk remain unchanged from previous iterations done




    def score_classification(clusters):
    # http://scikit-learn.org/stable/auto_examples/cluster/plot_adjusted_for_chance_measures.html
        """
        There are multiple ways to define what is close such as least squared loss function or Mahanalobis distance
        L least squares = (xi-xt)T * (xi-xj)
        L mahanalobis = (xi-xj)T * A(xi-xj)
        :param data: df of data to be clustered
        :param k_clusters: int number of clusters
        :param tolerance: float distance until convergence
        :param random_state: int seed of random generator
        :return: numpy array of clustered classes
        """

