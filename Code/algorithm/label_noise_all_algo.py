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
import time
from scipy.spatial.distance import cdist
from scipy import exp
from itertools import product
import csv
#
plot=0
max_itera=-1

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



def my_kernel(X, Y):
    """
    We create a custom kernel:
        should give 87% if it matches the original RBF kernel
    """
    S=0.84#parameter from rhos
    
    if dset==1:
        gamma= 0.0005   
    else:
        gamma= 0.00087 # maximise variance of kernel matrix        
    if np.array_equal(X,Y):
        N = X.shape[0]
        M=(1-S)*np.ones((N,N))+S*np.eye(N)
    else:
        M=1
    
    pairwise_sq_dists = cdist(X, Y, 'sqeuclidean')
    K = exp(-gamma*pairwise_sq_dists)*M
    return K

# dset chooses dataset. num_run determines the number of iterations.
def expectationMaximisation(run):    
    np.random.seed((run**5+1323002)%123123)#np.random.seed() alternatively
    print("dset:",dset,'run',run)
    Xtr,Str,Xts,Yts = data_cache[dset]
    X_train, X_val, y_train, y_val = train_test_split(Xtr, Str, test_size=prop)
    #clf1 is the first classifier while clf2 is the second
    clf = svm.SVC(C=2.5,kernel=my_kernel,max_iter=max_itera)
    if run==1:
        print("learn initial probability dset:",dset)  
    clf.fit(X_train,y_train)

    return clf.score(Xts,Yts)
    #23:08 23:12 23:28 4.2577 
def cv_reweighting(run):    
    np.random.seed((run**5+1323002)%123123)#np.random.seed() alternatively
    print("dset:",dset,'run',run)
    Xtr,Str,Xts,Yts = data_cache[dset]
    X_train, X_val, y_train, y_val = train_test_split(Xtr, Str, test_size=prop)
    #clf1 is the first classifier while clf2 is the second
    if dset==2:
        clf1 = svm.SVC(C=2.5,gamma=0.000225,probability=True,max_iter=max_itera)
    else:
        #removed 'gamma=scale'. should be the default.
        clf1 = svm.SVC(probability=True,gamma='scale',max_iter=max_itera)
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
        clf2 = svm.SVC(gamma=0.000225,C=0.8,max_iter=max_itera)
    else:
        clf2 = svm.SVC(gamma=0.00865,C=.4,max_iter=max_itera)

    clf2.fit(X_train,y_train,sample_weight=weights)
    #test_score=clf.score(Xts,Yts)
    #clf.score(Xtr,Str)
    # to accuracy 94.6 for dataset 1
    # 85.5 for dataset 2.
    return clf2.score(Xts,Yts)
    #23:08 23:12 23:28 4.2577 
    
def relabelling(run):    
    np.random.seed((run**5+1323002)%123123)#np.random.seed() alternatively
    print("dset:",dset,'run',run)
    Xtr,Str,Xts,Yts = data_cache[dset]
    X_train, X_val, y_train, y_val = train_test_split(Xtr, Str, test_size=prop)
    #clf1 is the first classifier while clf2 is the second
    if dset==2:
        clf1 = svm.SVC(C=2.5,gamma=0.000225,probability=True,max_iter=max_itera)
    else:
        clf1 = svm.SVC(gamma='scale',probability=True,max_iter=max_itera)
    if run==1:
        print("learn initial probability dset:",dset)    
    clf1.fit(X_train,y_train)   
    if run==1:
        print("calculating weighting dset:",dset)
    bb = clf1.predict_proba(X_train)
    print("estimate of [rho0 rho1]=",np.amin(bb, axis=0)) #rho0 rho1
    nn=len(y_train)
    ind=np.where(abs(bb[:,1]-y_train)>=0.5)
    y_train[ind]=1-y_train[ind]    
    ind_p=int(nn/3)
    ind5=np.hstack((np.argsort(-bb[:,1])[0:ind_p],np.argsort(-bb[:,0])[0:ind_p]))
    
    #The best parameters are {'C': 2.1544346900318834, 'gamma': 0.01} with a score of 1.00
#    search=0 #search for parameters
#    if search:
#        C_range = 1
#        gamma_range = np.logspace(10**-4,10**1, 4)
#        param_grid = dict(gamma=gamma_range, C=C_range)
#        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
#        grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv,n_jobs=-1)
#        grid.fit(Xtr[ind5,:],Str[ind5])
#        print("The best parameters are %s with a score of %0.2f"
#              % (grid.best_params_, grid.best_score_))    
    if dset==2:
        clf2 = svm.SVC(gamma=0.000225,max_iter=max_itera)
    else:
        clf2 = svm.SVC(gamma=0.00865,max_iter=max_itera)
    clf2.fit(X_train[ind5,:],y_train[ind5])
    return clf2.score(Xts,Yts)
    #23:08 23:12 23:28 4.2577 
    
def run_algorithm(alg_type, dset, num_run):   #alg_type: type of the algorithm, choose from 'reweighting',...tbc
    start=time.time()
    print('start of the whole algorithm with dataset',dset)
    pool = Pool(processes=cpu_count())
    if alg_type=='reweighting':
        print('start of reweighting algorithm')        
        it = pool.map(cv_reweighting, range(num_run))  #using the number of runs
        #test_score=np.zeros(num_run)
        #for i in range(num_run):
            #test_score[i]=cv_reweighting(1)
    if alg_type=='relabelling':
        print('start of relabelling algorithm')
        it = pool.map(relabelling, range(num_run))  #using the number of runs
    if alg_type=='expectationMaximisation':
        print('start of expectation Maximisation algorithm')
        it = pool.map(expectationMaximisation, range(num_run))  #using the number of runs
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
    with open('result'+'_data'+str(dset)+'_'+alg_type+str(round(end-start,4))+'sec.csv', 'w') as f: #better way to output result? I would like they can be read into python easily
        wr = csv.writer(f, dialect='excel')
        wr.writerows([test_score])
    
    print('total process time is',round(end-start,4),'sec')
    
    return average_score, std_score

average_score = {}
std_score={}
#TODO!please change it to for loop! for dset,algo in range(2),algos ...
for prop in np.linspace(0.8,0.,9):
    for dset, algo in product([2],['expectationMaximisation','relabelling','reweighting']):
        ind='dataset '+str(dset)+' '+algo
        average_score[ind],std_score[ind]=run_algorithm(algo,dset,16)
#dset=1
#average_score11, std_score11 = run_algorithm('reweighting',dset,16)
#average_score12, std_score12 = run_algorithm('relabelling',dset,16)
#average_score13, std_score13 = run_algorithm('expectationMaximisation',dset,16)
#dset=2
#average_score21, std_score21 = run_algorithm('reweighting',dset,16)
#average_score22, std_score22 = run_algorithm('relabelling',dset,16)
#average_score23, std_score23 = run_algorithm('expectationMaximisation',dset,16)
