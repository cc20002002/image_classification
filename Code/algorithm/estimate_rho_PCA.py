
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 17:52:27 2018

@author: chenc
Estimate rhos
results are 0.233 and 0.350 .
"""
import numpy as np

from sklearn.decomposition import IncrementalPCA as PCA

from densratio import densratio

dataset = np.load('../input_data/mnist_dataset.npz')


Xtr = dataset ['Xtr']
Str = dataset ['Str'].ravel()

pca = PCA(n_components=40)
pca.fit(Xtr)
Xtr=pca.transform(Xtr)
print(sum(pca.explained_variance_ratio_))

#There is a bug in the function compute_density_ratio that not allows us to search bandwidth more than 10.
#So scale the data so that the maximum bandwidth 10 will definitely oversmooth the principal components ranging in (-2000,3000).
Xtr=Xtr/100
PY1=sum(Str)/Str.shape
PY0=1-PY1
XY1=Xtr[Str==1,:]
XY0=Xtr[Str==0,:]
XY1oX=densratio(XY1,Xtr)
XY1oXV=min(XY1oX.compute_density_ratio(Xtr))
Y1X=XY1oXV*PY1
print(Y1X)

XY0oX=densratio(XY0,Xtr)
XY0oXV=min(XY1oX.compute_density_ratio(Xtr))
Y0X=XY0oXV*PY0
print(Y0X)