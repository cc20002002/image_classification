# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 18:56:00 2018

@author: chenc
"""
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
from sklearn.linear_model import LogisticRegression
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

clf = LogisticRegression(penalty='l1',C=0.015)
clf.fit(Xtr, Str) 
print(clf.score(Xtr, Str) )
print(clf.score(Xts, Yts) )
bb=clf.predict_proba(Xtr)
np.amin(bb, axis=0)

clf = RandomForestClassifier(n_estimators=100, max_depth=5)
