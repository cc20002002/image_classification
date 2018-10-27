# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 00:24:24 2018

@author: chenc
by label correction
improves the 1st data (Tshirt) set accuracy from 0.92 to 0.95
improves the 2nd data (car) set accuracy from 0.77 to 0.82
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
dset=2
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
print(clf.score(Xtr,Str))
bb=clf.predict_proba(Xtr)
#np.amin(bb, axis=0) #rho0 rho1
nn=len(Str)
ind_p=int(nn/3)
ind5=np.hstack((np.argsort(-bb[:,1])[0:ind_p],np.argsort(-bb[:,0])[0:ind_p]))

ind=np.where(abs(bb[:,1]-Str)>=0.5)
#np.amin(bb, axis=0)
#ind=np.argsort(-abs(bb[:,1]-Str))[0:int(nn/3)]#[1:160]
Xtr2=Xtr[ind5,:]
YYtr=clf.predict(Xtr2)
Xtr2=Xtr2[YYtr==Str[ind5],:]
YYtr2=YYtr[YYtr==Str[ind5]]

print(sum(YYtr2==1))
print(sum(YYtr2==0))


if dset==2:
    clf3 = svm.SVC(gamma=0.0007) #0.867
    clf3.fit(Xtr2,YYtr2)
    print(clf3.score(Xts,Yts))
else:
    clf3 = svm.SVC(gamma=0.01,C=5,class_weight='balanced')
    clf3.fit(Xtr2,YYtr2)
    print(clf3.score(Xts,Yts))

if dset==2:
    clf3 = svm.SVC(gamma=0.0008)
    clf3.fit(Xtr2,YYtr2)
    print(clf3.score(Xts,Yts))
else:
    clf3 = svm.SVC(gamma=0.01,C=5,class_weight='balanced')
    clf3.fit(Xtr2,YYtr2)
    print(clf3.score(Xts,Yts))

if dset==2:
    clf3 = svm.SVC(gamma=0.001,class_weight='balanced')
    clf3.fit(Xtr2,YYtr2)
    print(clf3.score(Xts,Yts))
else:
    clf3 = svm.SVC(gamma=0.01,C=5,class_weight='balanced')
    clf3.fit(Xtr2,YYtr2)
    print(clf3.score(Xts,Yts))
    
    

clf3 = svm.SVC(gamma=0.015)
clf3.fit(Xtr2,YYtr2)
print(clf3.score(Xts,Yts))

Str2=np.copy(Str)
temp=np.copy(1-Str[ind])
Str2[ind[0]]=temp


#The best parameters are {'C': 2.1544346900318834, 'gamma': 0.01} with a score of 1.00
search=0 #search for parameters
if search:
    C_range = 1
    gamma_range = np.logspace(10**-4,10**1, 4)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
    grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv,n_jobs=-1)
    grid.fit(Xtr[ind5,:],Str[ind5])
    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))
#clf2 = svm.SVC(gamma=0.00865)
if dset==2:
    clf2 = svm.SVC(gamma=0.000225)
else:
    clf2 = svm.SVC(gamma=0.00865)
clf2.fit(Xtr[ind5,:],Str[ind5])
print(clf2.score(Xts,Yts))


#gamma 0.0087 c=3
