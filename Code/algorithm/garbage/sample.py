"""
Created on Sat Oct 13 00:24:24 2018

@author: chenc
reweighting method from tut 2
dataset1 improves the accuracy  to 0.946
dataset1 improves the accuracy  to 0.855
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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
#from random import sample
dset=2
plot=0
n_estimators = 16
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

Xtr = dataset ['Xtr'].astype(float)
Str = dataset ['Str'].ravel()
Xts = dataset ['Xts'].astype(float)
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
    print('pca explained variance:',sum(pca.explained_variance_ratio_))
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

#indices = np.random.choice(Xts.shape[0],int(Xts.shape[0]*0.8), replace=False)

def estimateBeta(S,prob,rho0,rho1):
    S=S.astype(int)
    rho=np.array([rho1,rho0])
    #rho=np.tile(np.array([rho1, rho0]).reshape(-1,1),700).T
    #print(rho[S])
    
    #print(S)
    prob=prob[:,0]*(1-S[:])+prob[:,1]*(S[:])
    #print(sum(prob>.5)/700)
    #print(S[0:11])
    beta=(prob[:]-rho[S].ravel())/(1-rho0-rho1)/prob[:]
    return beta

# dset chooses dataset. num_run determines the number of iterations.
def cv_reweighting(dset, num_run):
    if dset==1:
        clf = svm.SVC(C=.8,gamma=0.000225,probability=True)
        print('running for fashion_mnist')
    else:
        #removed 'gamma=scale'. should be the default.
        clf = svm.SVC(probability=True,C=.4,gamma=0.00865)
        print('running for cifar')
    
    val_score=np.zeros(num_run)
    
    for run in range(num_run):
        X_train, X_val, y_train, y_val = train_test_split(Xtr, Str, test_size=0.2)

        clf.fit(X_train,y_train)
        #print(clf.score(Xts,Yts))
        #clf.score(Xtr,Str)
        probS = clf.predict_proba(X_train)
        weights = estimateBeta(y_train, probS, 0.2, 0.4)
        #print(weights.shape)

        for i in range(len(weights)):
            if weights[i] < 0:
                weights[i] = 0.0    
                
        if dset==2:
            clf = svm.SVC(gamma=0.000225/5**2,C=0.8,probability=True)
        else:
            clf = svm.SVC(gamma=0.00865/5**2,C=.4,probability=True)
        clf=BaggingClassifier(clf, max_samples=1.0 / 5, n_estimators=n_estimators,n_jobs =-1)  
        clf.fit(X_train,y_train,sample_weight=weights)
        val_score[run]=clf.score(Xts,Yts)
        print('run ',run,' validation score after reweighting:',val_score[run])
        #clf.score(Xtr,Str)
        # to accuracy 94.6 for dataset 1
        # 85.5 for dataset 2.
    average_score=np.mean(val_score)
    print('average score: ',average_score)
    return average_score, clf


average_score,clf=cv_reweighting(dset, 1)
#average_score,clf2=cv_reweighting(2, 1)

X_train, X_val, y_train, y_val = train_test_split(Xtr, Str, test_size=0.2)
temp=10000 / (2 * np.bincount(y_train)/np.array((1.2,0.8)))
temp={0:temp[0],1:temp[1]}
clf=BaggingClassifier(svm.SVC(probability=True,C=.4,gamma=0.0004\
                              ,class_weight=temp)\
    , max_samples=1.0 / 5, n_estimators=n_estimators,n_jobs =-1)
clf.fit(X_train,y_train)
clf.score(Xts,Yts)
probS = clf.predict_proba(X_train)
weights = estimateBeta(y_train, probS, 0.2, 0.4)
#print(weights.shape)

for i in range(len(weights)):
    if weights[i] < 0:
        weights[i] = 0.0    
        
if dset==2:
    clf = svm.SVC(gamma=0.0006,C=0.7,probability=True,class_weight=temp)
else:
    clf = svm.SVC(gamma=0.00865,C=.4,probability=True,class_weight=temp)
clf=BaggingClassifier(clf, max_samples=1.0 / 5, n_estimators=n_estimators,n_jobs =-1)  
clf.fit(X_train,y_train,sample_weight=weights)
clf.score(Xts,Yts)
print("score for dataset dset: ",clf.score(Xts,Yts)) #yutong please include dset variable in this print

