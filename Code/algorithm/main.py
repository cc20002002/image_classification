"""
Sat Oct 13 00:24:24 2018
@author: chenc TinyC
Please install sklearn 0.20.0
Plase install latest version of multiprocessing,numpy and matplotlib

"""


import numpy as np
from sklearn import svm
from os import cpu_count
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import time
from scipy.spatial.distance import cdist
from scipy import exp
from itertools import product
from util import estimateBeta, load_data
import csv


# Global settings

# 1 = MINIST, 2=CIFAR
dset = 1

# test dataset split rate
prop = 0.2

# Maximum of iteration. When testing the algorithm, set it to be small like 100
max_itera = -1

# load the data into data_cache. Use `dset` on the top to change the dataset.
# -- data_cache[1] for MINIST
# -- data_cache[2] for CIFAR.
data_cache = load_data()


def my_kernel(X, Y):
    """
    We create a custom kernel. This custom kernel is used by the function expectationMaximisation.
    Please see Section 3.5.3 for the mathematical derivation.
    Parameters
    ----------
    X,Y: two vectors or two matrix.

    Returns
    ----------
    clf.score(Xts, Yts): the kernel product of vectors X and Y.
    """
    S = 0.84  # parameter from rhos

    if dset == 1:
        gamma = 0.0005
    else:
        gamma = 0.00087  # maximise variance of kernel matrix
    if np.array_equal(X, Y):
        N = X.shape[0]
        M = (1 - S) * np.ones((N, N)) + S * np.eye(N)
    else:
        M = 1

    pairwise_sq_dists = cdist(X, Y, 'sqeuclidean')
    K = exp(-gamma * pairwise_sq_dists) * M
    return K


# dset chooses dataset.
# num_run determines the number of iterations.
def expectationMaximisation(run):
    """
    This function implements the expectation Maximisation model classifying 
    image with label noise described in Section 3.5.
   
    Parameters
    ----------
    Run: a seed number.

    Returns
    ----------

    clf.score(Xts, Yts): the accuracy of the algorithm on test data
    """
    print('run = ', run)
    # np.random.seed() alternatively
    # customise the seed number tahe way you want
    np.random.seed((run ** 5 + 1323002) % 123123)  
    print("dset:", dset, 'run', run)
    Xtr, Str, Xts, Yts = data_cache[dset]
    X_train, X_val, y_train, y_val = train_test_split(Xtr, Str, test_size=prop)
    # clf1 is the first classifier while clf2 is the second
    clf = svm.SVC(C=2.5, kernel=my_kernel, max_iter=max_itera)
    if run == 1:
        print("learn initial probability dset:", dset)
    clf.fit(X_train, y_train)

    return clf.score(Xts, Yts)
    # 23:08 23:12 23:28 4.2577


def cv_reweighting(run):
    """
    This function implements the Importance reweighting model classifying image 
    with label noise described in Section 3.6.
   
    Parameters
    ----------
    Run: a seed number.

    Returns
    ----------

    clf.score(Xts, Yts): the accuracy of the algorithm on test data
    """
    np.random.seed((run ** 5 + 1323002) % 123123)  # np.random.seed() alternatively
    print("dset:", dset, 'run', run)

    Xtr, Str, Xts, Yts = data_cache[dset]
    X_train, X_val, y_train, y_val = train_test_split(Xtr, Str, test_size=prop)

    # clf1 is the first classifier while clf2 is the second
    if dset == 2:
        clf1 = svm.SVC(C=2.5, gamma=0.000225, probability=True, max_iter=max_itera)
    else:
        # removed 'gamma=scale'. should be the default.
        clf1 = svm.SVC(probability=True, gamma='scale', max_iter=max_itera)

    if run == 1:
        print("learn initial probability dset:", dset)
    clf1.fit(X_train, y_train)

    if run == 1:
        print("calculating weighting dset:", dset)

    probS = clf1.predict_proba(X_train)
    weights = estimateBeta(y_train, probS, 0.2, 0.4)

    for i in range(len(weights)):
        if weights[i] < 0:
            weights[i] = 0.0
    if run == 1:
        print("calculating final model dset:", dset)
    if dset == 2:
        clf2 = svm.SVC(gamma=0.000225, C=0.8, max_iter=max_itera)
    else:
        clf2 = svm.SVC(gamma=0.00865, C=.4, max_iter=max_itera)

    clf2.fit(X_train, y_train, sample_weight=weights)

    return clf2.score(Xts, Yts)


def relabelling(run):
    """
    This function implements the heuristic approach of classifying image with 
    label noise described in Section 3.7.
   
    Parameters
    ----------
    Run: a seed number.

    Returns
    ----------

    clf.score(Xts, Yts): the accuracy of the algorithm on test data
    """
    np.random.seed((run ** 5 + 1323002) % 123123)  # np.random.seed() alternatively
    print("dset:", dset, 'run', run)
    Xtr, Str, Xts, Yts = data_cache[dset]
    X_train, X_val, y_train, y_val = train_test_split(Xtr, Str, test_size=prop)
    # clf1 is the first classifier while clf2 is the second
    if dset == 2:
        clf1 = svm.SVC(C=2.5, gamma=0.000225, probability=True, max_iter=max_itera)
    else:
        clf1 = svm.SVC(gamma='scale', probability=True, max_iter=max_itera)
    if run == 1:
        print("learn initial probability dset:", dset)
    clf1.fit(X_train, y_train)
    if run == 1:
        print("calculating weighting dset:", dset)
    bb = clf1.predict_proba(X_train)
    nn = len(y_train)
    ind = np.where(abs(bb[:, 1] - y_train) >= 0.5)
    y_train[ind] = 1 - y_train[ind]
    ind_p = int(nn / 3)
    ind5 = np.hstack((np.argsort(-bb[:, 1])[0:ind_p], np.argsort(-bb[:, 0])[0:ind_p]))

    if dset == 2:
        clf2 = svm.SVC(gamma=0.000225, max_iter=max_itera)
    else:
        clf2 = svm.SVC(gamma=0.00865, max_iter=max_itera)

    clf2.fit(X_train[ind5, :], y_train[ind5])
    return clf2.score(Xts, Yts)


def run_algorithm(alg_type, dset, num_run):
    # alg_type: type of the algorithm, choose from 'reweighting',...tbc
    start = time.time()
    print('start of the whole algorithm with dataset', dset)
    pool = Pool(processes=cpu_count())

    if alg_type == 'reweighting':
        print('start of reweighting algorithm')
        it = pool.map(cv_reweighting, range(num_run), )  # using the number of runs

    if alg_type == 'relabelling':
        print('start of relabelling algorithm')
        it = pool.map(relabelling, range(num_run))  # using the number of runs

    if alg_type == 'expectationMaximisation':
        print('start of expectation Maximisation algorithm')
        print('num_run', num_run)
        it = pool.map(expectationMaximisation, range(num_run))  # using the number of runs

    pool.close()
    pool.join()
    test_score = it
    average_score = np.mean(test_score)
    std_score = np.std(test_score)
    print('average score: ', average_score, '\nstandard deviation: ', std_score)  # help to format here!
    end = time.time()
    with open('result'+str(prop) + '_data' + str(dset) + '_' + alg_type + str(round(end - start, 4)) + 'sec.csv',
              'w') as f:  # better way to output result? I would like they can be read into python easily
        wr = csv.writer(f, dialect='excel')
        wr.writerows([test_score])

    print('total process time is', round(end - start, 4), 'sec')

    return average_score, std_score


def main():
    """Run Three different Algorithm on dataset MINIST or CIFAR"""

    # initialise result dictionaries
    average_score = {}
    std_score = {}
    for dset, algo in product([1, 2], ['expectationMaximisation', 'relabelling', 'reweighting']):
        ind = 'dataset ' + str(dset) + ' ' + algo
        print(ind)
        average_score[ind], std_score[ind] = run_algorithm(algo, dset, 16)


if __name__ == '__main__':
    main()
