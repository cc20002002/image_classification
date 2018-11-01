import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA as PCA


def load_data(dset):
    # result stores the data splits
    result = {}

    # Load the image dataset and input image parameters
    dataset1 = np.load('../input_data/mnist_dataset.npz')
    dataset2 = np.load('../input_data/cifar_dataset.npz')

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
    result[1] = (Xtr1, Str1, Xts1, Yts1)

    Xts2 = scaler.fit_transform(Xts2.T).T
    Xtr2 = scaler.fit_transform(Xtr2.T).T
    if dset==2:   
    # principal component analysis for dataset 2
        pca = PCA(n_components=100)
        pca.fit(Xtr2)
        Xtr2 = pca.transform(Xtr2)
        Xts2 = pca.transform(Xts2)
        result[2] = (Xtr2, Str2, Xts2, Yts2)

    return result


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