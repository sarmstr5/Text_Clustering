# General packages
from pylab import *
import pandas as pd
import numpy as np
import sys
from datetime import datetime as dt
from itertools import cycle
# For plotting
import matplotlib.pylab as plt
# For Matrices
from scipy.sparse import csc_matrix, coo_matrix
# Data Science Packages
from sklearn.model_selection import cross_val_score, StratifiedKFold #cross validation packages
from sklearn.decomposition import TruncatedSVD
# from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import silhouette_score, calinski_harabaz_score, make_scorer
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


# N_PARAMETERS = 126373 #based on features.txt

############Data Visualization##################
#needs work
def plot_csc(csc):
    plt.spy(csc, aspect='auto')
    plt.show()

def plot_coo(csc):
    if not isinstance(csc, coo_matrix):
        m = coo_matrix(csc)
    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg='black')
    ax.plot(m.col, m.row, 's', color='white', ms=1)
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('auto')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.figure.show()

def plot_PCA(components, scores):
    plt.figure()
    plt.plot(components, scores)

############General Methods###############
def get_time():
    time = dt.now()
    hour, minute = str(time.hour), str(time.minute)
    if (len(minute) == 1):
        minute = '0' + minute
    if (len(hour) == 1):
        hour = '0' + hour
    time = hour + minute
    return time

########## Metrics ##############
def s_scoring(y, y_predicted):
    # F1 = 2 * (precision * recall) / (precision + recall), harmonic mean of precision and recall
    return f1_score(y, y_predicted, average='None')  # returns list score [pos neg], can use weighted

########## Cross Validation Methods ############
def run_svd_cross_validation_runs(x, folds, min_n, max_n, steps, k_clusters, verbose):
    if verbose:
        print('Running SVD Cross Validation:\n initial n: {}\t final n: {}\t tims: {}'.format(min_n, max_n, get_time()))
    fn = 'test_output/' + "cross_validation_SVD_results" + '.txt'
    n = min_n
    feature_selection_method = 'SVD'
    while n < max_n:
        if verbose:
            print('n_components: {}\t time: {}'.format(n, get_time()))
        x_svd, svd_variance = feature_selection(x, n, verbose)  # is this sparse?
        evaluate_models(x_svd, k_clusters, folds, fn, feature_selection_method, n, svd_variance, verbose)
        n += steps

def run_param_cross_validation_runs(x, folds, min_k, max_k, steps, feature_selection_method, n_params, verbose):
    if verbose:
        print('Running Parameter Cross Validation: {}'.format(get_time()))
    fn = 'test_output/' + "cross_validation_param_results" + '.txt'
    x_svd, svd_variance = feature_selection(x, n_params, verbose)  # is this sparse?
    k = min_k
    while k < max_k:
        evaluate_models(x_svd, k, folds, fn, feature_selection_method, n_params, svd_variance, verbose)
        k += steps

def evaluate_models(x, k, folds, fn, feature_select, n_params, variance_explained, verbose):
    if verbose:
        print('In evaluate model: {}'.format(get_time()))

    km_model = get_kmeans(x, k, verbose=verbose)
    km_x = km_model.fit(x)
    x_labels = km_x.labels_

    if verbose:
        print('Getting score: {}'.format(get_time()))
        print('shape of x: {} shape of labels: {}'.format(x.shape, x_labels.shape))
        print(x_labels)
    # scorer = make_scorer(silhouette_score, metric='cosine')
    score = cross_val_score(km_model, x, cv=folds).mean()
    score_metric = 'SSE'
    cosine = False
    # if score_metric == 'silhouette':
    #     score = silhouette_score(x, x_labels, metric='euclidean')
    # else:
    #     score = calinski_harabaz_score(x, x_labels)
    if verbose:
        print('Score: {}\t time: '.format(score, get_time()))
    with open(fn, 'a+') as csv:
        csv.write("{0}\t{1}\t{2}\t{3}\t{4}\t\t{5}\t\t{6}\t\t{7}\t\t{8}\n".format(
            get_time(), round(score, 2), k, folds, feature_select, n_params, round(variance_explained, 2), score_metric, cosine))

############# Modeling Methods ################
def feature_selection(x, svd_n, verbose):
    if verbose:
        print('Performing SVD: {}'.format(get_time()))
    # ~100% of variance is explained by 800 variables
    svd_model = TruncatedSVD(algorithm='randomized', n_components=svd_n)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd_model, normalizer)
    x_svd = lsa.fit_transform(x)
    explained_variance = svd_model.explained_variance_ratio_.sum()

    if verbose:
        print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))
    return x_svd, explained_variance

def predict_test_set_svd(n_params, test, x, y, x_pos, y_pos, error):
    svd_m = decomposition.TruncatedSVD(algorithm='randomized', n_components=n_params, n_iter=7)
    svm_m = svm.SVC(kernel='linear', C=error, probability=True, random_state=random_state)  # C is penalty of error
    # SVM/SVD with positive set
    # first get fit truncated SVD on positive data set, trim full and test parameters
    svd_model = svd_m.fit(x_pos, y_pos)
    x_svd = svd_model.transform(x)
    # now train svm on tranformed data sets
    svm_model = svm_m.fit(x_svd, y)
    # predict test set
    svd_svm_predictions = svm_model.predict(test_svd)

    return svd_svm_predictions, svd_svm_test_prob, svd_svm_train_prob

def get_kmeans(x, k_clusters=8, n_rand_runs=10, prec_loops=300, tolerance=1e-5, verbose=False):
    if verbose:
        print('Getting kmeans model: {}'.format(get_time()))
    kmeans_model = KMeans(n_clusters=k_clusters, n_init=n_rand_runs, max_iter=prec_loops, tol=tolerance, verbose=False, n_jobs=-1)
    return kmeans_model

###########IO methods#############
def data_fn(get_full_dataset, verbose):
    if verbose:
        print("Getting Filenames, Full Dataset Run: {}".format(get_full_dataset))

    dir = 'data/'
    if (get_full_dataset == True):
        features_fn = dir + 'features.txt'
        csc_fn = dir + "input_csc.npz"
        articles_fn = dir + "input_text"
    else:
        features_fn = dir + 'features.txt'
        csc_fn = dir + "input_short_csc.npz"
        articles_fn = dir + "input_short_text"
    return features_fn, csc_fn, articles_fn

def read_in_data(features_fn, articles_csc_fn, articles_dense_fn, get_txt_dense, verbose):
    if verbose:
        print("Reading in Data: {}".format(get_time()))
    npy_file = np.load(articles_csc_fn)
    csc = csc_matrix((npy_file['data'], npy_file['indices'], npy_file['indptr']), shape=npy_file['shape'])
    dict = pd.read_csv(features_fn, header=None, names=['words'])
    if get_txt_dense:
        articles_df = pd.read_csv(articles_dense_fn, header=None)
    else:
        articles_df = None
    return dict, csc, articles_df

def get_processed_data(get_full_dataset, get_txt_dense, verbose):
    if verbose:
        print("Getting processed cluster data: {}".format(get_time()))
    features_fn, articles_csc_fn, articles_dense_fn = data_fn(get_full_dataset, verbose)
    features, articles_csc, articles_df = read_in_data(features_fn, articles_csc_fn, articles_dense_fn, get_txt_dense,
                                                       verbose)
    return features, articles_csc, articles_df

def clusters_to_csv(labels):
    test_output = 'cluster_output/test_results_silhouette{}.csv'.format(get_time())
    with open(test_output, 'w') as results:
        for y in labels:
            results.write('{0}\n'.format(y))

if __name__ == '__main__':
    # Initial Conditions
    verbose = True              # print out steps
    get_full_dataset = True    # use truncated or full data set
    get_articles_dense = False  # get large text file
    cv_feature_selection = True # perform cross validation on feature selection
    cv_model_params = False      # perform cross validation on model params
    do_data_viz = False         # create graphs at various steps
    do_print_results = False    # print output to file

    print('Starting Cross Validation Run at {}'.format(get_time()))

    if verbose:
        print('Cross Validation Conditions\n get_full_dataset: {}\n get_articles_dense: {}\n cv_feature_selection: {}\n'
              ' cv_model_params: {}\n do_data_viz: {}\n do_print_results : {}'
              .format(get_full_dataset, get_articles_dense, cv_feature_selection, cv_model_params, do_data_viz,
                      do_print_results))

    if verbose:
        print('Reading in Processed Data: {}'.format(get_time()))
    features, x, x_dense = get_processed_data(get_full_dataset, get_articles_dense, verbose)

    if do_data_viz:
        if verbose:
            print('Visualizing sparse matrices: {}'.format(get_time()))
        plot_coo(x_csc)
        plt.spy(x_csc, aspect='auto')
        plt.show()

    if cv_feature_selection:
        if verbose:
            print('Performing Feature Selection: {}'.format(get_time()))
        # converts the word frequencies into floats neg and pos
        folds = 3
        params = 200
        max_params = 600
        steps = 25
        k_clusters = 7
        run_svd_cross_validation_runs(x, folds, params, max_params, steps, k_clusters, verbose)

        #need to do
        if do_data_viz:
            plot_PCA(components, scores)
    else:
        # SVM performs best with SVD reduction, at error rate = .93 with 22 params
        svd_n_params = 22
        if verbose:
            print('Using n_params: {} \t : {}'.format(param, get_time()))

    if verbose:
        print('Clustering Data: {}'.format(get_time()))

    if cv_model_params:
        # Cross Validation Params
        folds = 10
        min_k = 2
        max_k = 50
        steps = 2
        n_params = 1000
        run_param_cross_validation_runs(x, folds, min_k, max_k, steps, 'SVD', n_params, verbose)

    else:
        pass
        # Instantiating model parameters
        # k_clusters = 5
        # x_km = get_kmeans(x, k_clusters)
        # x_labels = x_km.predict(x)

    # Print Results
    if do_print_results:
        clusters_to_csv(x_labels)

    print('Ending Cross Validation Run at {}'.format(get_time()))
    print('-----------FINISHED-------------')

