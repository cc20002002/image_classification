"""
Created on Sat Oct 13 00:24:24 2018
Please install sklearn 0.20.0
Plase install latest version of multiprocessing,numpy and matplotlib
@author: chenc TinyC
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
#from sklearn.model_selection import StratifiedShuffleSplit
#from sklearn.model_selection import GridSearchCV
from sklearn import svm
from os import cpu_count
from sklearn.decomposition import IncrementalPCA as PCA
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
#from random import sample


#
plot=0


dataset1 = np.load('../input_data/mnist_dataset.npz')
size_image1 = 28
dim_image1=1 
dataset2 = np.load('../input_data/cifar_dataset.npz')
size_image2 =32
dim_image2 =3
#size_image=28
#dim_image=1

#to store the data splits
data_cache={}

#transform dataset1
Xtr1 = dataset1 ['Xtr'].astype(float)
Str1 = dataset1 ['Str'].ravel()
Xts1 = dataset1 ['Xts'].astype(float)
Yts1 = dataset1 ['Yts'].ravel()
scaler = StandardScaler()
Xts1 = scaler.fit_transform(Xts1.T).T
Xtr1 = scaler.fit_transform(Xtr1.T).T
data_cache[1]=(Xtr1,Str1,Xts1,Yts1)

#transform dataset2
Xtr2 = dataset2 ['Xtr'].astype(float)
Str2 = dataset2 ['Str'].ravel()
Xts2 = dataset2 ['Xts'].astype(float)
Yts2 = dataset2 ['Yts'].ravel()
#scaler = StandardScaler()
Xts2 = scaler.fit_transform(Xts2.T).T
Xtr2 = scaler.fit_transform(Xtr2.T).T
data_cache[2]=(Xtr2,Str2,Xts2,Yts2)

#Xtr=Xtr.reshape(10000,dim_image,size_image,size_image).transpose([0,2, 3, 1]).mean(3).reshape(10000,size_image*size_image)
#Xts=Xts.reshape(2000,dim_image,size_image,size_image).transpose([0,2, 3, 1]).mean(3).reshape(2000,size_image*size_image)
pca = PCA(n_components=100)
pca.fit(Xtr2)
Xtr2=pca.transform(Xtr2)
Xts2=pca.transform(Xts2)
print('pca explained variance:',sum(pca.explained_variance_ratio_))
if plot:
    xplot=scaler.fit_transform(pca.inverse_transform(Xts2).T).T
##1plt.jet()
dset=1
if plot:
    plt.figure()
    for i in range(0,30):
        if dset==1:
            image=xplot[i,].reshape(dim_image1,size_image1,size_image1).transpose([1, 2, 0])
            plt.subplot(5, 6, i+1)
            plt.imshow(image[:,:,:],interpolation='bicubic')
            plt.title(Yts1[i])
        else:
            image=xplot[i,].reshape(dim_image2,size_image2,size_image2).transpose([1, 2, 0])
            plt.subplot(5, 6, i+1)
            plt.imshow(image[:,:,:],interpolation='bicubic')
            plt.title(Yts2[i])

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



import time
# dset chooses dataset. num_run determines the number of iterations.
def cv_reweighting(run):    
    np.random.seed((run**5+1323002)%123123)#np.random.seed() alternatively
    print("dset:",dset,'run',run)
    Xtr,Str,Xts,Yts = data_cache[dset]
    X_train, X_val, y_train, y_val = train_test_split(Xtr, Str, test_size=0.2)
    #clf1 is the first classifier while clf2 is the second
    if dset==2:
        clf1 = svm.SVC(C=2.5,gamma=0.000225,probability=True)
    else:
        #removed 'gamma=scale'. should be the default.
        clf1 = svm.SVC(probability=True,gamma=0.00865)
    if run==1:
        print("learn initial probability dset:",dset)        
    clf1.fit(X_train,y_train)
    #print(clf.score(Xts,Yts))
    #clf.score(Xtr,Str)
    if run==1:
        print("calculating weighting dset:",dset)
    probS = clf1.predict_proba(X_train)
    weights = estimateBeta(y_train, probS, 0.2, 0.4)
    #print(weights.shape)

    for i in range(len(weights)):
        if weights[i] < 0:
            weights[i] = 0.0    
    if run==1:
        print("calculating final model dset:",dset)
    if dset==2:
        clf2 = svm.SVC(gamma=0.000225,C=0.8)
    else:
        clf2 = svm.SVC(gamma=0.00865,C=.4)

    clf2.fit(X_train,y_train,sample_weight=weights)
    #test_score=clf.score(Xts,Yts)
    #clf.score(Xtr,Str)
    # to accuracy 94.6 for dataset 1
    # 85.5 for dataset 2.
    return clf2.score(Xts,Yts)
    #23:08 23:12 23:28 4.2577 
def run_algorithm(alg_type, dset, num_run):   #alg_type: type of the algorithm, choose from 'reweighting',...tbc
    start=time.time()
    print('start of the whole algorithm with dataset',dset)
    if alg_type=='reweighting':
        print('start of reweighting algorithm')
        pool = Pool(processes=cpu_count())
        it = pool.map(cv_reweighting, range(num_run))  #using the number of runs
        #test_score=np.zeros(num_run)
        #for i in range(num_run):
            #test_score[i]=cv_reweighting(1)
    pool.close()
    pool.join()
    test_score= it
    average_score=np.mean(test_score)
    std_score=np.std(test_score)
    print('average score: ',average_score,'\nstandard deviation: ',std_score) # help to format here!
    end=time.time()
    with open('result'+'_data'+str(dset)+'_'+alg_type+'.txt', 'w') as f: #better way to output result? I would like they can be read into python easily
        for item in test_score:
            f.write("%s\n" % item)
    
    print('total process time is',round(end-start,4),'sec')
    
    return average_score, std_score
#change it to for loop?
dset=1
average_score1, std_score1 = run_algorithm('reweighting',dset,16)
dset=2
average_score2, std_score2 = run_algorithm('reweighting',dset,16)