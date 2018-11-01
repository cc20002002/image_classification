import numpy as np
from scipy.spatial.distance import cdist
from scipy import exp
from sklearn import svm
from sklearn.model_selection import train_test_split

def estimateBeta(S, prob, rho0, rho1):
    """
    This function was estimated use method proposed by Liu and Tao.

    Parameters
    ----------
    S is the training labels with noise
    prob is the conditional probability predicted by a pretraining model.
    described in Section 3.4 in our report.
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
    We desinged a new kernel for the function expectationMaximisation.
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
    This function implements the Expectation Maximisation algorithm classifying 
    image with label noise described in Section 3.5.
   
    Parameters
    ----------
    Run: a seed number.

    Returns
    ----------

    clf.score(Xts, Yts): the accuracy of the algorithm on test data
    """
    # np.random.seed() alternatively
    # customise the seed number tahe way you want
    np.random.seed((run ** 5 + 1323002) % 123123)  

    Xtr, Str, Xts, Yts = data_cache[dset]
    X_train, X_val, y_train, y_val = train_test_split(Xtr, Str, test_size=prop)
    # clf1 is the first classifier while clf2 is the second
    clf = svm.SVC(C=2.5, kernel=my_kernel, max_iter=max_itera)
    if run == 1:
        print("learn probability dset:", dset)
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
    

    Xtr, Str, Xts, Yts = data_cache[dset]
    X_train, X_val, y_train, y_val = train_test_split(Xtr, Str, test_size=prop)

    # clf1 is the first classifier while clf2 is the second
    if dset == 2:
        clf1 = svm.SVC(C=2.5, gamma=0.000225, probability=True, max_iter=max_itera)
    else:
        clf1 = svm.SVC(gamma = 'scale',probability=True, max_iter=max_itera)
    if run == 1:
        print("learn initial probability dset:", dset)
    clf1.fit(X_train, y_train)
    return clf1.score(Xts, Yts)
    if run == 1:
        print("calculating weighting dset:", dset)

    probS = clf1.predict_proba(X_train)
    weights = estimateBeta(y_train, probS, 0.2, 0.4)

    for i in range(len(weights)):
        if weights[i] < 0:
            weights[i] = 0.0
    if run == 1:
        print("fit final model dset:", dset)
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

    Xtr, Str, Xts, Yts = data_cache[dset]
    X_train, X_val, y_train, y_val = train_test_split(Xtr, Str, test_size=prop)
    # clf1 is the first classifier while clf2 is the second
    if dset == 2:
        clf1 = svm.SVC(C=2.5, gamma=0.000225, probability=True, max_iter=max_itera)
    else:
        clf1 = svm.SVC(gammma = 'scale',probability=True, max_iter=max_itera)
    if run == 1:
        print("learn pre training model:")
    clf1.fit(X_train, y_train)
    if run == 1:
        print("calculating weighting and fit final model:")
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