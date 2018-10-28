"""
Sat Oct 13 00:24:24 2018
@author: chenc TinyC
Please install sklearn 0.20.0
Plase install latest version of multiprocessing,numpy and matplotlib

"""


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from os import cpu_count
from sklearn.decomposition import IncrementalPCA as PCA
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import time
from scipy.spatial.distance import cdist
from scipy import exp
from itertools import product
import csv


def estimateBeta(S, prob, rho0, rho1):
    """
    This function was estimated use method proposed by (???)
  
    Parameters
    ----------
    S is the training labels with noise
    prob is the conditional probability predicted by a pretraining model. 
    described in equation ?? in our report.
    rho0, rho1 are the flip rates.

    Returns
    ----------
    beta:  the Importance weighting for the second training model.
    Parameters.
    """
    
    S = S.astype(int)
    rho = np.array([rho1, rho0])
    prob = prob[:, 0] * (1 - S[:]) + prob[:, 1] * (S[:])
    beta = (prob[:] - rho[S].ravel()) / (1 - rho0 - rho1) / prob[:]
    return beta

def my_kernel(X, Y):
    """
    We create a custom kernel. This custom kernel is used by the function expectationMaximisation.
    Please see Section *** for the mathematical derivation.
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

# dset chooses dataset. num_run determines the number of iterations.
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
    # np.random.seed() alternatively
    # customise the seed number the way you want
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
    # print(clf.score(Xts,Yts))
    # clf.score(Xtr,Str)
    if run == 1:
        print("calculating weighting dset:", dset)
    probS = clf1.predict_proba(X_train)
    weights = estimateBeta(y_train, probS, 0.2, 0.4)
    # print(weights.shape)

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

#    if search:
#        C_range = 1
#        gamma_range = np.logspace(10**-4,10**1, 4)
#        param_grid = dict(gamma=gamma_range, C=C_range)
#        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
#        grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv,n_jobs=-1)
#        grid.fit(Xtr[ind5,:],Str[ind5])
#        print("The best parameters are %s with a score of %0.2f"
#              % (grid.best_params_, grid.best_score_))

def run_algorithm(alg_type, dset, num_run):  # alg_type: type of the algorithm, choose from 'reweighting',...tbc
    start = time.time()
    print('start of the whole algorithm with dataset', dset)
    pool = Pool(processes=cpu_count())
    if alg_type == 'reweighting':
        print('start of reweighting algorithm')
        it = pool.map(cv_reweighting, range(num_run))  # using the number of runs

    if alg_type == 'relabelling':
        print('start of relabelling algorithm')
        it = pool.map(relabelling, range(num_run))  # using the number of runs
    if alg_type == 'expectationMaximisation':
        print('start of expectation Maximisation algorithm')
        it = pool.map(expectationMaximisation, range(num_run))  # using the number of runs

    pool.close()
    pool.join()
    test_score = it
    average_score = np.mean(test_score)
    std_score = np.std(test_score)
    print('average score: ', average_score, '\nstandard deviation: ', std_score)  # help to format here!
    end = time.time()
    with open(str(prop) + '_data' + str(dset) + '_' + alg_type + str(round(end - start, 4)) + 'sec.csv',
              'w') as f:  # better way to output result? I would like they can be read into python easily
        wr = csv.writer(f, dialect='excel')
        wr.writerows([test_score])

    print('total process time is', round(end - start, 4), 'sec')

    return average_score, std_score


#Maximum of iteration. When testing the algorithm, set it to be small like 100
max_itera = -1


#Load the image dataset and input image parameters
dataset1 = np.load('../input_data/mnist_dataset.npz')
dataset2 = np.load('../input_data/cifar_dataset.npz')
size_image1 = 28
dim_image1 = 1
size_image2 = 32
dim_image2 = 3

# data_cache stores the data splits
data_cache = {}

# transform dataset to appropriate size
Xtr1 = dataset1['Xtr'].astype(float)
Str1 = dataset1['Str'].ravel()
Xts1 = dataset1['Xts'].astype(float)
Yts1 = dataset1['Yts'].ravel()

Xtr2 = dataset2['Xtr'].astype(float)
Str2 = dataset2['Str'].ravel()
Xts2 = dataset2['Xts'].astype(float)
Yts2 = dataset2['Yts'].ravel()

# Standardise images 
scaler = StandardScaler()
Xts1 = scaler.fit_transform(Xts1.T).T
Xtr1 = scaler.fit_transform(Xtr1.T).T
data_cache[1] = (Xtr1, Str1, Xts1, Yts1)

Xts2 = scaler.fit_transform(Xts2.T).T
Xtr2 = scaler.fit_transform(Xtr2.T).T

#principal component analysis for dataset 2
pca = PCA(n_components=100)
pca.fit(Xtr2)
Xtr2 = pca.transform(Xtr2)
Xts2 = pca.transform(Xts2)
data_cache[2] = (Xtr2, Str2, Xts2, Yts2)
#print('pca explained variance:', sum(pca.explained_variance_ratio_))

#initialise result dictionaries
average_score = {}
std_score = {}
prop=0.2
for prop in np.linspace(0.8,0.2,7):
    for dset, algo in product([1, 2], ['expectationMaximisation', 'relabelling', 'reweighting']):
        ind = 'dataset ' + str(dset) + ' ' + algo
        average_score[ind], std_score[ind] = run_algorithm(algo, dset, cpu_count())
