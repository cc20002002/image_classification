# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from random import sample
class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
    
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
Xtr = scaler.fit_transform(Xtr)

search=0 #search for parameters
if search:
    C_range = np.logspace(-1, 3, 4)
    gamma_range = np.logspace(-4, 1, 4)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
    grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
    grid.fit(Xts, Yts)
    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))

indices = np.random.choice(Xts.shape[0], 
                           int(Xts.shape[0]*0.8), replace=False)


clf = svm.SVC(gamma='scale',probability=True)
Ytsc=np.hstack((1-Yts[0:534],Yts[534:1600]))
# idx = np.argpartition(bb,10)

clf.fit(Xts[0:1600,:],Ytsc.ravel())

clf.score(Xts[1600:2000,:],Yts[1600:2000].ravel())

#aa=clf.decision_function(Xts[0:1600,:])
bb=clf.predict_proba(Xts[0:1600,:])
#aa=abs(aa)
#aaidx=np.argsort(aa)
#aa[aaidx]
#idx2=np.argsort(abs(bb[:,1]-0.5))
#bb[idx2,1]
ind=np.argsort(-abs(bb[:,1]-Ytsc))[0:170]#[1:160]
threshold=abs(bb[:,1]-Ytsc)[ind][-1]
Ytsc[ind]=1-Ytsc[ind]

clf2 = svm.SVC(gamma='scale',probability=True)
clf2.fit(Xts[0:1600,:],Ytsc.ravel())
clf2.score(Xts[1600:2000,:],Yts[1600:2000].ravel())
bb2=clf.predict_proba(Xts[0:1600,:])
ind=np.argsort(-abs(bb2[:,1]-Ytsc))[0:80]
Ytsc[ind]=1-Ytsc[ind]

clf2.fit(Xts[0:1600,:],Ytsc.ravel())
clf2.score(Xts[1600:2000,:],Yts[1600:2000].ravel())
bb2=clf.predict_proba(Xts[0:1600,:])
ind=np.argsort(-abs(bb2[:,1]-Ytsc))[0:50]
np.argsort(-abs(bb2[:,1]-Ytsc))<534
Ytsc[ind]=1-Ytsc[ind]

clf2.fit(Xts[0:1600,:],Ytsc.ravel())
clf2.score(Xts[1600:2000,:],Yts[1600:2000].ravel())
bb2=clf.predict_proba(Xts[0:1600,:])
ind=np.argsort(-abs(bb2[:,1]-Ytsc))[0:30]
Ytsc[ind]=1-Ytsc[ind]
print(abs((Ytsc.astype(np.float32)-Yts[0:1600].astype(np.float32))).sum())

for i in range(5):
    clf2.fit(Xts[0:1600,:],Ytsc.ravel())
    ss=clf2.score(Xts[1600:2000,:],Yts[1600:2000].ravel())
    print(ss)
    bb2=clf.predict_proba(Xts[0:1600,:])
    ind=np.argsort(-abs(bb2[:,1]-Ytsc))[0:10]
    ind=np.where(abs(abs(bb2[:,1]-Ytsc)-0.5)<0.01)[0]
    ind=ind[np.random.randint(2, size=len(ind)).astype(bool)]
    print(np.sort(-abs(bb2[:,1]-Ytsc))[0:10])
    print(ind)
    Ytsc[ind]=1-Ytsc[ind]
    print(abs((Ytsc.astype(np.float32)-Yts[0:1600].astype(np.float32))).sum())

np.argsort(-abs(bb2[:,1]-Ytsc))<534
tt=np.cumsum(np.argsort(-abs(bb2[:,1]-Ytsc))<534)/np.linspace(1,1600,1600)
plt.figure();plt.plot(tt)
Xts.shape
clf.n_support_

import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))
Xts[:,1].mean()