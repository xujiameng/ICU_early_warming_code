# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:09:18 2018

@author: 佳盟
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:39:13 2018

@author: 佳盟
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 10:16:34 2018

@author: dell
"""

import pandas as pd
import time
from sklearn import cross_validation
import numpy as np
import random
from numpy import *
#调用分类器
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb    

import lightgbm as lgb
from sklearn import ensemble
#from GCForest import gcForest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

#评价指标
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef # MCC
from sklearn.metrics import confusion_matrix #混淆矩阵
from sklearn.preprocessing import StandardScaler
from  imblearn.under_sampling  import RandomUnderSampler
from  imblearn.over_sampling  import RandomOverSampler
from sklearn.grid_search import GridSearchCV

# 传统评分用
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import scipy.io as sio  
from sklearn.metrics import confusion_matrix  
from sklearn import metrics
import itertools
import copy

from sklearn.model_selection import StratifiedKFold

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.constraints import maxnorm





# 提取测试集，训练集
#comtest= pd.read_csv("new_0712_pat_with_characteristic_cat_allrunin_complete_in_rf_part_dim_reduction_onehot.csv")

comtest = pd.read_csv("0904_16g_3kinds_order.csv")
tt=151;
z=[]
for i in range(tt+1):
    z.append(i)
z.append(comtest.shape[1]-1)
comtest = comtest.iloc[:,z]


scaler = StandardScaler()
comtest.iloc[:,0:comtest.shape[1]-1] = scaler.fit_transform(comtest.iloc[:,0:comtest.shape[1]-1])



def evaluating_indicator(y_true, y_test, y_test_value):
    c_m = confusion_matrix(y_true, y_test)
    TP=c_m[0,0]
    FN=c_m[0,1]
    FP=c_m[1,0]
    TN=c_m[1,1]
    
    TPR=TP/ (TP+ FN) #敏感性
    TNR= TN / (FP + TN) #特异性
    BER=1/2*((FP / (FP + TN) )+FN/(FN+TP))
    
    ACC = accuracy_score(y_true, y_test)
    MCC = matthews_corrcoef(y_true, y_test)
    F1score =  f1_score(y_true, y_test)
    AUC = roc_auc_score(y_true,y_test_value[:,0])
    
    c={"TPR" : TPR,"TNR" : TNR,"BER" : BER
    ,"ACC" : ACC,"MCC" : MCC,"F1_score" : F1score,"AUC" : AUC}
    return c



def blo(pro_comm_Pre,jj):
    blo_Pre=zeros(len(pro_comm_Pre))
    blo_Pre[(pro_comm_Pre[:,0]>(jj*0.01))]=1
    return blo_Pre




def RUN():
    tiaocan_train, ceshi_train, tiaocan_train_test, ceshi_true = cross_validation.train_test_split(comtest.iloc[0:len(comtest),1:comtest.shape[1]-1],comtest.iloc[0:len(comtest),-1], test_size = 0.2,random_state = 0)    
    position=[];
    skf=StratifiedKFold(n_splits=10)
#    for times in range(10):
#    tiaocan_train = comtest.iloc[:,0:comtest.shape[1]-1]
#    tiaocan_train_test = comtest.iloc[:,-1]
    tiaocan_train=np.array(tiaocan_train,dtype=np.float16)
    tiaocan_train_test=np.array(tiaocan_train_test,dtype=np.float16)
    times=0
    position=[]
    for train, test in skf.split(tiaocan_train,tiaocan_train_test):
        alltime_start=time.time()
        times=times+1


        x_train=tiaocan_train[train]
        y_train=tiaocan_train_test[train]
        x_test=tiaocan_train[test]
        y_true=tiaocan_train_test[test]        
#        x_train, y_train = RandomUnderSampler().fit_sample(x_train, y_train)

        comm = Sequential()
        comm.add(Dropout(0.2, input_shape=(tt,)))
        comm.add(Dense(int(100), init='normal', activation='sigmoid', W_constraint=maxnorm(3)))     
        comm.add(Dense(int(50), init='normal', activation='sigmoid', W_constraint=maxnorm(3))) 
 

        comm.add(Dense(1, init='normal', activation='sigmoid'))
        sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
        comm.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        comm.fit(x_train, y_train, nb_epoch=100, batch_size=8000)
        pro_comm_Pre = comm.predict_proba(x_test)

############################ 最大MCC ###################################################
#        fpr, tpr, thresholds = roc_curve(y_true, pro_comm_Pre[:,1], pos_label=1)
#        RightIndex=(tpr+(1-fpr)-1)
#        positon=np.argmax(RightIndex)
#        aw=int(positon)   
#        th=thresholds[aw];
#        position.append(th)
#        print('done_0, 第%s次验证 '%(times)) 
#    position=np.array(position,dtype=np.float16)
######################## 敏感性特异性相近 ###############################################
        RightIndex=[]
        for jj in range(100):
            blo_comm_Pre = blo(pro_comm_Pre,jj)
            eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre)
            RightIndex.append(abs(eva_comm['TPR'] - eva_comm['TNR']))
        RightIndex=np.array(RightIndex,dtype=np.float16)
        position=np.argmin(RightIndex)
        
        alltime_end=time.time()
        print('done_0, 第%s次验证, time:%s s '%(times,alltime_end-alltime_start)) 
#        print('time:%s s'%(alltime_end-alltime_start)) 
######################################################################################
    return  position.mean()
best_th = RUN()


#######################################################
#def RUN():
#    
#    tiaocan_train, ceshi_train, tiaocan_train_test, ceshi_true = cross_validation.train_test_split(comtest.iloc[0:len(comtest),1:comtest.shape[1]-1],comtest.iloc[0:len(comtest),-1], test_size = 0.2,random_state = 0)    
#    position=[];
#    x_train = tiaocan_train
#    y_train = tiaocan_train_test
#    x_test = ceshi_train
#    y_true = ceshi_true
#    
#    x_train=np.array(x_train,dtype=np.float16)
#    y_train=np.array(y_train,dtype=np.float16)
#    x_test=np.array(x_test,dtype=np.float16)
#    y_true=np.array(y_true,dtype=np.float16)
##    x_train, y_train = RandomUnderSampler().fit_sample(x_train, y_train)
#    
#    
#    comm=xgb.XGBClassifier(learning_rate =0.1, n_estimators=500, max_depth=5, min_child_weight=5,n_jobs=8,
#                                tree_method='exact',objective = 'rank:pairwise',
#                                colsample_bytree=0.8, reg_alpha=0.005)
#    comm.fit(x_train , y_train)
#    pro_comm_Pre = comm.predict_proba(x_test)
#    
#######################################################################################
#    fpr, tpr, thresholds = roc_curve(y_true, pro_comm_Pre[:,1], pos_label=1)
#    RightIndex=(tpr+(1-fpr)-1)
#    positon=np.argmax(RightIndex)
#    aw=int(positon)   
#    th=thresholds[aw];
#    position.append(th) 
#    position=np.array(position,dtype=np.float16)
#######################################################################################
##    RightIndex=[]
##    for jj in range(100):
##        blo_comm_Pre = blo(pro_comm_Pre,jj)
##        eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre)
##        RightIndex.append(abs(eva_comm['TPR'] - eva_comm['TNR']))
##    RightIndex=np.array(RightIndex,dtype=np.float16)
##    position=np.argmin(RightIndex)
#
#######################################################################################
#
#    return  position
#best_th = RUN()
########################################################


print(' done_1  best_th')

def RUN_2(best_th):
    comm_s_TPR=[];comm_s_TNR=[];comm_s_BER=[];comm_s_ACC=[];comm_s_MCC=[];comm_s_F1score=[];comm_s_AUC=[];comm_s_time=[];
    tiaocan_train, ceshi_train, tiaocan_train_test, ceshi_true = cross_validation.train_test_split(comtest.iloc[0:len(comtest),1:comtest.shape[1]-1],comtest.iloc[0:len(comtest),-1], test_size = 0.2,random_state = 0)    


    x_train = tiaocan_train
    y_train = tiaocan_train_test
    x_test = ceshi_train
    y_true = ceshi_true
    
    
    x_train=np.array(x_train,dtype=np.float16)
    y_train=np.array(y_train,dtype=np.float16)
    x_test=np.array(x_test,dtype=np.float16)
    y_true=np.array(y_true,dtype=np.float16)
#    x_train, y_train = RandomUnderSampler().fit_sample(x_train, y_train)
    
    comm = Sequential()
    comm.add(Dropout(0.2, input_shape=(tt,)))
    
    comm.add(Dense(int(100), init='normal', activation='sigmoid', W_constraint=maxnorm(3))) 
    comm.add(Dense(int(50), init='normal', activation='sigmoid', W_constraint=maxnorm(3))) 

        
    comm.add(Dense(1, init='normal', activation='sigmoid'))
    sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
    comm.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    comm.fit(x_train, y_train, nb_epoch=100, batch_size=8000)
    
    pro_comm_Pre = comm.predict_proba(x_test)
    blo_comm_Pre = blo(pro_comm_Pre,best_th)
    eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre)
 
    comm_s_TPR.append(eva_comm['TPR']);comm_s_TNR.append(eva_comm['TNR']);comm_s_BER.append(eva_comm['BER']);
    comm_s_ACC.append(eva_comm['ACC']);comm_s_MCC.append(eva_comm['MCC']);comm_s_F1score.append(eva_comm['F1_score']);
    comm_s_AUC.append(eva_comm['AUC']);
    eva_comm={"TPR" : np.mean(comm_s_TPR),"TNR" : np.mean(comm_s_TNR),"BER" :  np.mean(comm_s_BER)
    ,"ACC" : np.mean(comm_s_ACC),"MCC" : np.mean(comm_s_MCC),"F1_score" : np.mean(comm_s_F1score)
    ,"AUC" : np.mean(comm_s_AUC),"time" : np.mean(comm_s_time)}    
    
    return  eva_comm
eva_comm_best = RUN_2(best_th)

print(' done_2  eva_comm_best')

#eva_comm_50 = RUN_2(50)
#
#print(' done_3  eva_comm_50')


