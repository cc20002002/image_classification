# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 11:22:18 2018

@author: chenc
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 00:24:24 2018

@author: chenc
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from scipy.spatial.distance import cdist
from scipy import exp
from cvxopt import blas
from cvxopt.base import matrix
#from random import sample
'''
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
'''    
os.chdir('C:/Users/chenc/Documents/GitHub/image_classification/Code/input_data')
dataset = np.load('mnist_dataset.npz')
Xtr = dataset ['Xtr']
Str = dataset ['Str'].ravel()
Xts = dataset ['Xts']
Yts = dataset ['Yts'].ravel()
plt.gray()
plt.figure()
for i in range(0,30):
    image=Xts[i,].reshape(28,28)
    plt.subplot(5, 6, i+1)
    plt.imshow(image)
    plt.title(Yts[i])
Y=Yts
Xts.shape
scaler = StandardScaler()
Xts = scaler.fit_transform(Xts.T).T
Xtr = scaler.fit_transform(Xtr.T).T
S=0.84#parameter from rhos

indices = np.random.choice(Xts.shape[0], 
                           int(Xts.shape[0]*0.8), replace=False)
def my_kernel(X, Y):
    """
    We create a custom kernel:
        should give 87% if it matches the original RBF kernel
    """
    
    if np.array_equal(X,Y):
        N = X.shape[0]
        M=(1-S)*np.ones((N,N))+S*np.eye(N)
    else:
        M=1
    
    pairwise_sq_dists = cdist(X, Y, 'sqeuclidean')
    K = exp(-0.0087*pairwise_sq_dists)*M
    #sigma=(2*0.0087)**(-0.5)
    #X = matrix(X)
    #N,n = X.size
    #Q = matrix(0.0, (N,N))
    #ones = matrix(1.0, (N,1))
    #blas.syrk(X, Q, alpha = 1.0/sigma)
    #a = Q[::N+1]
    #blas.syr2(a, ones, Q, alpha = -0.5)  
    #Q = exp(Q)
    return K

clf = svm.SVC(kernel=my_kernel)
clf.fit(Xtr,Str)
clf.score(Xts,Yts)



