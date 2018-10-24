#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 18:25:18 2018

@author: chenc
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 17:52:27 2018

@author: chenc
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import IncrementalPCA as PCA
#from random import sample
from densratio import densratio
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

PY1=sum(Str)/Str.shape
PY0=1-PY1
XY1=Xtr[Str==1,:]
XY0=Xtr[Str==0,:]
XY1oX=densratio(XY1,Xtr)
XY1oXV=min(XY1oX.compute_density_ratio(Xtr))
Y1X=XY1oXV*PY1
print(Y1X)

XY0oX=densratio(XY0,Xtr)
XY0oXV=min(XY0oX.compute_density_ratio(np.random.uniform(Xtr.min()*.95,Xtr.max()*.95,[1000,Xtr.shape[1]])))
Y0X=XY0oXV*PY0
print(Y0X)