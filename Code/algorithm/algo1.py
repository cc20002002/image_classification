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
dataset.files
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

search=1 #search for parameters


indices = np.random.choice(Xts.shape[0], 
                           int(Xts.shape[0]*0.8), replace=False)


clf = svm.SVC(gamma='scale',probability=True)
clf.fit(Xtr,Str)
clf.score(Xts,Yts)
bb=clf.predict_proba(Xtr)
np.amin(bb, axis=0) #rho0 rho1
nn=len(Str)
ind=np.where(abs(bb[:,1]-Str)>=0.5)
np.amin(bb, axis=0)
#ind=np.argsort(-abs(bb[:,1]-Str))[0:int(nn/3)]#[1:160]
Str[ind]=1-Str[ind]

ind_p=int(10000/3)
ind5=np.hstack((np.argsort(-bb[:,1])[0:ind_p],np.argsort(-bb[:,0])[0:ind_p]))
if search:
    C_range = np.logspace(-1, 3, 4)
    gamma_range = np.logspace(-4, 1, 4)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
    grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
    grid.fit(Xtr[ind5,:],Str[ind5])
    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))
clf2 = svm.SVC(gamma='scale')
clf2.fit(Xtr[ind5,:],Str[ind5])
clf2.score(Xts,Yts)


