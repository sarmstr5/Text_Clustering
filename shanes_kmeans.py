import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn import cluster
from sklearn import preprocessing
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
    inertia = 0

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

    def initial_cluster_locs(data):
        k_clusters = self.obj_d['k_clusters']
        have_a_median_cluster = random.choice([True, False])
        min_js = self.obj_d['min_js']
        max_js = self.obj_d['max_js']
        initial_cluster_locs = [[0]*data.columns.max() for k in range(k_clusters)] # k by j 0 matrix
        for k in range(k_clusters):
            if(k == 0 and have_a_median_cluster):
                for j in data.columns:
                    initial_cluster_locs[k][j] = (min_js[j] + max_js[j])/2  # may not be necessary calc if normalized
            for j in data.columns:
                initial_cluster_locs[k][j] = random.randrange(min_js[j], max_js[j])
            return initial_cluster_locs

    def find_min_k(cluster means, xi):
        for
    def partition_slices(end, n_partitions):
        slices = [slice(i, i + n_partitions) for i in range(0, end, n_partitions)]
        return slices

    def cluster_rows(X, slice_i, cluster_means):

    def predict(self, X):
        """
        Predict cluster for xi
        :param X: data to be predicted
        :return numpy array labels
        """
        data = self.obj_d['data']
        #0 initialize random cluster means
        cluster_locs = initial_cluster_locs(data)
        converged = False
        slices = partition_slices(len(data), cpu_count())   #list indices for multiprocessing

        while(not converged):
            for row in data.values.tolist():
                #1 for xi find k that minimizes D, closest cluster mean, set Znk =1, Znj = 0
                pool = Pool()
                results = pool.map(find_min_k, (cluster_locs, row))

                #2 If all of assignments Znk remain unchanged from previous iterations
                #3 Update cluster means uk
                #4 Start over



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

