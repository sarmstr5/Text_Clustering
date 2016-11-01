from pylab import *
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from sklearn import preprocessing
import itertools

MAX_PARAMS = 126373 #based on features.txt
def file_to_list(file, verbose=False):
    if verbose:
        print("converting training data to list")
    file = 'data/' + file
    max_param_count = 0
    param_list = []
    with open(file, 'r') as csv:
        param_num = 0
        bag_of_indexes = []
        word_frequency_list = []
        word_freq_by_article = [0]*MAX_PARAMS
        for row in csv:
            # each row is are word indexes paired with their counts, all separated by spaces
            row_list = row.split(' ')
            row_list = [int(i) for i in row_list]   # convert to numbers from split
            words = slice(0, len(row_list), 2)
            counts = slice(1, len(row_list), 2)
            word_indexes = [n-1 for n in row_list[words]] # subtract one to correct for 0 index
            word_count = row_list[counts]
            for i in word_indexes:
                word_freq_by_article[i] += 1

            # each appended item is a article with its word (reference indexes) and respective counts
            # creates jagged list with variying length reviews
            bag_of_indexes.append(word_indexes)
            word_frequency_list.append(word_count)

    return bag_of_indexes, word_frequency_list, word_freq_by_article

def reduce_words()
def jagged_list_to_csc(index_list, word_freqs, verbose):
    if verbose:
        print("in jagged to csc method")
    i = 0
    param_lists = []
    row_lists = []
    value_lists = []
    max_params = 0
    for indices, frequencies in zip(index_list, word_freqs):
        max_index = max(indices)
        row = [i] * len(indices)  # list of row number with length of indexes
        param_lists.append(indices)
        row_lists.append(row)
        value_lists.append(frequencies)
        i += 1
        if (max_index > max_params):
            max_params = max_index  # words used are parameters
        if not (len(indices) == len(frequencies) and len(frequencies) == len(row)):
            print("----------ERROR SOMETHING NOT RIGHT, ARTICLE {}".format(i))
    if verbose:
        print("maxcolumn: {}".format(max_params))
    coo = create_coo(param_lists, row_lists, value_lists, i, max_params, verbose)
    return csr_matrix(coo)

#helper method creates sparse COOrdinate format matrix for fast construction
def create_coo(param_lists, row_lists, value_lists, num_rows, num_cols, verbose):
    if verbose:
        print('Creating Sparse CSR Bag of words')
        print("rows{0} \t cols: {1}".format(num_rows, num_cols))
    # in create CSR
    flattened_params = np.array(list(itertools.chain.from_iterable(param_lists)))
    flattened_rows = np.array(list(itertools.chain.from_iterable(row_lists)))
    flattened_values = np.array(list(itertools.chain.from_iterable(value_lists)))
    sparse_coo = coo_matrix((flattened_values, (flattened_rows, flattened_params)),  # three 1D lists
                            shape=(num_rows, num_cols+1),   #cant remember why index is plus 1.  why not -1?
                            dtype=np.int8)  # creates a int compressed sparse row matrix

    return sparse_coo
# t = training data, x = test data
def write_csc_to_disk(csc, fn, verbose):
    if verbose:
        print('csc file {0} to disk'.format(fn))
    fn = 'data/' + fn
    np.savez(fn, data=csc.data, indices=csc.indices, indptr=csc.indptr, shape=csc.shape)

def write_list_to_disk(dense_bag_of_words, fn, verbose):
    if verbose:
        print("Writing articles to list {}".format(fn))
    fn = 'data/' + fn
    with open(fn, 'w') as csv:
        for row in dense_bag_of_words:
            csv.write("{0}\n".format(row))

def words_to_list(file, verbose):
    if verbose:
        print("converting feature file to list of words")
    file = 'data/' + file
    words_list = []
    with open(file, 'r') as csv:
        word_list = [word.strip() for word in csv]
    return word_list

def create_articles(word_list, index_csc, verbose):
    if verbose:
        print("Converting Sparse index matrix to articles")
    bag_of_indexes = index_csc.toarray()
    bag_of_articles = ['']*bag_of_indexes.shape[0]  #number of rows
    article_i = 0
    for article in bag_of_indexes:
        words = [] #assuming appending saves space and time instead of instantiating
        i = 0
        for word_count in article:
            word= ''
            if(word_count > 0):
                words.append(str(word_list[i]+' ')*int(word_count)) # multiple by word count for repeated words
            i += 1
        bag_of_articles[article_i] = ("".join(words))    # convert list of words to string append to bag
        article_i += 1
    return bag_of_articles

def main():
    verbose = True
    full_run = True
    create_articles_file = False
    if verbose:
        print('Starting File')
    if full_run:
        input_fn = 'input.txt'
        features_fn = 'features.txt'
        csc_fn = "input_csc"
        articles_fn = "output_text"
    else:
        input_fn = 'input_short.txt'
        features_fn = 'features.txt'
        csc_fn = "input_short_csc"
        articles_fn = "input_short_text"

    # convert data to lists
    if verbose:
        print("convert data to lists")
    word_jagged_list, word_freq, word_freq_by_article = file_to_list(input_fn, verbose)
    feature_list = words_to_list(features_fn, verbose)

    # convert lists to cscs
    if verbose:
        print("converting index and frequency lists to csc and article lists")
    bag_of_indexes_csc = jagged_list_to_csc(word_jagged_list, word_freq, verbose)
    if create_articles_file:
        articles = create_articles(feature_list, bag_of_indexes_csc, verbose)

    # write cleaned data to file
    if verbose:
        print("Writing processed information to file")
    print(bag_of_indexes_csc[:5])
    write_csc_to_disk(bag_of_indexes_csc, csc_fn, verbose)
    if create_articles_file:
        write_list_to_disk(articles, articles_fn, verbose)

    if verbose:
        print("------------FINISHED-----------")

if __name__ == '__main__':
    main()
