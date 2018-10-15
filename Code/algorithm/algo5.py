#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 13:32:36 2018

@author: chenc
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 00:24:24 2018

@author: chenc
improves the accuracy from 0.8395 to 0.95
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.decomposition import IncrementalPCA as PCA
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
dset=1
plot=0

if dset==1:
    dataset = np.load('../input_data/mnist_dataset.npz')
    size_image=28
    dim_image=1 
else:
    dataset = np.load('../input_data/cifar_dataset.npz')
    size_image=32
    dim_image=3
#size_image=28
#dim_image=1

Xtr = dataset ['Xtr']
Str = dataset ['Str'].ravel()
Xts = dataset ['Xts']
Yts = dataset ['Yts'].ravel()
scaler = StandardScaler()
Xts = scaler.fit_transform(Xts.T).T
Xtr = scaler.fit_transform(Xtr.T).T

if dset==2:
    #Xtr=Xtr.reshape(10000,dim_image,size_image,size_image).transpose([0,2, 3, 1]).mean(3).reshape(10000,size_image*size_image)
    #Xts=Xts.reshape(2000,dim_image,size_image,size_image).transpose([0,2, 3, 1]).mean(3).reshape(2000,size_image*size_image)
    pca = PCA(n_components=100)
    pca.fit(Xtr)
    Xtr=pca.transform(Xtr)
    Xts=pca.transform(Xts)
    print(sum(pca.explained_variance_ratio_))
    if plot:
        xplot=scaler.fit_transform(pca.inverse_transform(Xts).T).T

#Xts = scaler.fit_transform(Xts.T).T
#Xtr = scaler.fit_transform(Xtr.T).T

##1plt.jet()

if plot:
    plt.figure()
    for i in range(0,30):
        image=xplot[i,].reshape(dim_image,size_image,size_image).transpose([1, 2, 0])
        plt.subplot(5, 6, i+1)
        plt.imshow(image[:,:,:],interpolation='bicubic')
        plt.title(Yts[i])

indices = np.random.choice(Xts.shape[0], 
                           int(Xts.shape[0]*0.8), replace=False)

if dset==2:
    clf = svm.SVC(C=2.5,gamma=0.000225,probability=True)
else:
    clf = svm.SVC(gamma='scale',probability=True)
    
  

clf.fit(Xtr,Str)
print(clf.score(Xts,Yts))
clf.score(Xtr,Str)

def estimateBeta(S,prob,rho0,rho1):
    S=S.astype(int)
    rho=np.array([rho1,rho0])
    #rho=np.tile(np.array([rho1, rho0]).reshape(-1,1),700).T
    #print(rho[S])
    
    #print(S)
    prob=prob[:,0]*(1-S[:])+prob[:,1]*(S[:])
    print(sum(prob>.5)/700)
    #print(S[0:11])
    beta=(prob[:]-rho[S].ravel())/(1-rho0-rho1)/prob[:]
    return beta

probS = clf.predict_proba(Xtr)
#print(Str[0:10])

#print(rho.shape)
#rho[:,Str.astype(int)]

#print(probS.shape)
weights = estimateBeta(Str, probS, 0.2, 0.4)
print(weights.shape)
# remove negative weights and normalize weights to 1
for i in range(len(weights)):
    if weights[i] < 0:
        weights[i] = 0.0    
if dset==2:
    clf = svm.SVC(gamma=0.000225,C=.8)
else:
    clf = svm.SVC(gamma='scale',C=.4)
clf.fit(Xtr,Str,sample_weight=weights)
print(clf.score(Xts,Yts))
clf.score(Xtr,Str)
