# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 11:22:18 2018

@author: chenc
improves the accuracy from 0.8395 to 0.9235 0.9355 0.946
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
#from sklearn.model_selection import StratifiedShuffleSplit
#from sklearn.model_selection import GridSearchCV
from sklearn import svm
from scipy.spatial.distance import cdist
from scipy import exp
#from cvxopt import blas
#from cvxopt.base import matrix
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

#dataset = np.load('mnist_dataset.npz')
dataset = np.load('../input_data/cifar_dataset.npz')
#size_image=28
#dim_image=1
size_image=32
dim_image=3
Xtr = dataset ['Xtr']
Str = dataset ['Str'].ravel()
Xts = dataset ['Xts']
Yts = dataset ['Yts'].ravel()
#plt.gray()
plt.figure()
for i in range(0,30):
    image=Xts[i,].reshape(size_image,size_image,dim_image)
    plt.subplot(5, 6, i+1)
    plt.imshow(image[:,:,1])
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
        M=10
    
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
# error on the training data and minimising the norm of the weights. It is analageous to the ridge parameter in ridge regression (in fact in practice there is little difference in performance or theory between linear SVMs and ridge regression, so I generally use the latter - or kernel ridge regression if there are more attributes than observations).
#For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a better job of getting all the training points classified correctly.
clf = svm.SVC(kernel=my_kernel)
clf.fit(Xtr,Str)
print(clf.score(Xtr,Str))
clf.score(Xts,Yts)



