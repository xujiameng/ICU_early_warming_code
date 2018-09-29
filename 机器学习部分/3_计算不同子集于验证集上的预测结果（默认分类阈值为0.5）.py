

# Description 
# 程序功能：计算不同机器学习模型不同特征子集下，验证集上的预测结果评价指标
# 程序流程：
#       Step1：对不同特征子集进行交叉验证
#       Step2：计算交叉验证结果评价指标的平均值与标准差并以CSV格式输出
# 程序运行结果：输出预测结果评价指标的平均值与标准差
#
# DataFile: 病例为1*N的向量
#   0808_16g_3kinds_order.csv 进行特征排序后的数据集
#
# Output:
#   mean    预测结果评价指标的平均值
#   std     预测结果评价指标的标准差
# V1.0 2018/9/13


import pandas as pd
import time
from sklearn import cross_validation
import numpy as np
import random
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






# 提取原始数据集，只包含病例特征及衍生变量，第一列为subject_id，最后一列为death
comtest= pd.read_csv("0808_16g_3kinds_order.csv")
  
scaler = StandardScaler()
comtest.iloc[:,0:comtest.shape[1]-1] = scaler.fit_transform(comtest.iloc[:,0:comtest.shape[1]-1]) #对特征数据进行标准化



def blo(pro_comm_Pre):     #默认分类阈值为0.5，并根据预测概率值预测患者生死情况
    blo_Pre=[];
    for i in range(len(pro_comm_Pre)):
        if pro_comm_Pre[i,1]>0.5:
            blo_Pre.append(1)
        else:
                blo_Pre.append(0)
    return blo_Pre




def evaluating_indicator(y_true, y_test, y_test_value):  #计算预测结果的各项评价指标
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
    AUC = roc_auc_score(y_true,y_test_value[:,1])
    
    c={"TPR" : TPR,"TNR" : TNR,"BER" : BER
    ,"ACC" : ACC,"MCC" : MCC,"F1_score" : F1score,"AUC" : AUC}
    return c



def RUN(jj):  #主函数
    #为预测结果的评价指标提供存储空间
    comm_s_TPR=[];comm_s_TNR=[];comm_s_BER=[];comm_s_ACC=[];comm_s_MCC=[];comm_s_F1score=[];comm_s_AUC=[]; 
   
    tiaocan_train, ceshi_train, tiaocan_train_test, ceshi_true = cross_validation.train_test_split(comtest.iloc[0:len(comtest),1:comtest.shape[1]-1],comtest.iloc[0:len(comtest),-1], test_size = 0.2,random_state = 0)      #提取病例数据的特征
  

    skf=StratifiedKFold(n_splits=10)   #设置十折交叉验证

    tiaocan_train=np.array(tiaocan_train,dtype=np.float16)
    tiaocan_train_test=np.array(tiaocan_train_test,dtype=np.float16)
    times=0
    for train, test in skf.split(tiaocan_train,tiaocan_train_test):
        alltime_start=time.time()

        times=times+1

        x_train=tiaocan_train[train]
        y_train=tiaocan_train_test[train]
        x_test=tiaocan_train[test]
        y_true=tiaocan_train_test[test]

        x_train, y_train = RandomUnderSampler().fit_sample(x_train, y_train)   #对训练集进行类平衡，平衡方法为对样本较多的生还患者进行欠采样
        x_train = x_train[:,0:jj]   #取前jj个特征  
        x_test = x_test[:,0:jj]  
    
    
        # 设置机器学习模型
##########################################################################################################################
##########################################################################################################################
        
############################# --regr-- #############################
        comm =RandomForestClassifier(n_estimators=5000, criterion='entropy', oob_score=True,min_samples_split=10,
                                      random_state=10,  min_samples_leaf=10,n_jobs=-1,warm_start=True)
        
############################## --XGB-- #############################
#        comm = xgb.XGBClassifier(learning_rate =0.01, n_estimators=5000, max_depth=8, min_child_weight=8,n_jobs=8,
#                                tree_method='exact',objective = 'rank:pairwise',
#                                colsample_bytree=0.8, reg_alpha=0.005)
#        
############################## --GBM-- #############################
#        comm = lgb.LGBMClassifier(max_bin = 255,
#                                 num_leaves=1000,learning_rate =0.01,n_estimators=5000,n_jobs=-1,
#                                 reg_alpha=0.08,max_depth=8, min_child_weight=6)  
#        
############################## --Adb-- #############################
#        comm = ensemble.AdaBoostClassifier(learning_rate =0.1, n_estimators=500)
#        
############################## --regr-- #############################
#        comm = svm.SVC(C=0.999, probability=True) 
#        
############################## --Log-- #############################
#        comm = LogisticRegression(solver='saga',warm_start=True)
#        
############################## --GBN-- #############################
#        a=0.01
#        comm = GNB=GaussianNB(priors=[a,1-a])
#        
############################## --DT-- #############################
#        comm = DecisionTreeClassifier(splitter='random',min_samples_split=20,
#                                  min_samples_leaf=80 ,max_leaf_nodes=None)
 
##########################################################################################################################  
##########################################################################################################################
        comm.fit(x_train , y_train)   #对模型进行训练
        pro_comm_Pre = comm.predict_proba(x_test)  #对验证集样本进行预测
        blo_comm_Pre = blo(pro_comm_Pre)   #根据预测概率得出患者生死预测的结果
        eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre)  #计算预测结果的各项评价指标
        alltime_end=time.time()
        print('done_gbm 第%s次验证, time: %s s '%(times alltime_end-alltime_start))
        
        # 储存每次预测结果的评价指标
        comm_s_BER.append(eva_comm['BER']);
        comm_s_TPR.append(eva_comm['TPR']);
        comm_s_TNR.append(eva_comm['TNR']);
        comm_s_ACC.append(eva_comm['ACC']);
        comm_s_MCC.append(eva_comm['MCC']);
        comm_s_F1score.append(eva_comm['F1_score']);
        comm_s_AUC.append(eva_comm['AUC']);
        
    mean_BER_score=[];std_BER_score=[];
    mean_TPR_score=[];std_TPR_score=[];
    mean_TNR_score=[];std_TNR_score=[];
    mean_ACC_score=[];std_ACC_score=[];
    mean_MCC_score=[];std_MCC_score=[];
    mean_F1score_score=[];std_F1score_score=[];        
    mean_AUC_score=[];std_AUC_score=[];    
        
    #计算交叉验证结果的平均值与标准差
    mean_BER_score=np.mean(comm_s_BER);std_BER_score=np.std(comm_s_BER);
    mean_TPR_score=np.mean(comm_s_TPR);std_TPR_score=np.std(comm_s_TPR);
    mean_TNR_score=np.mean(comm_s_TNR);std_TNR_score=np.std(comm_s_TNR);
    mean_ACC_score=np.mean(comm_s_ACC);std_ACC_score=np.std(comm_s_ACC);
    mean_MCC_score=np.mean(comm_s_MCC);std_MCC_score=np.std(comm_s_MCC);
    mean_F1score_score=np.mean(comm_s_F1score);std_F1score_score=np.std(comm_s_F1score);        
    mean_AUC_score=np.mean(comm_s_AUC);std_AUC_score=np.std(comm_s_AUC);
        
        
    return mean_BER_score,std_BER_score,  mean_TPR_score,std_TPR_score,  mean_TNR_score,std_TNR_score,  mean_ACC_score,std_ACC_score,  mean_MCC_score,std_MCC_score,  mean_F1score_score,std_F1score_score,  mean_AUC_score,std_AUC_score

meanBERfit=[];stdBERfit=[];
meanTPRfit=[];stdTPRfit=[];
meanTNRfit=[];stdTNRfit=[];
meanACCfit=[];stdACCfit=[];
meanMCCfit=[];stdMCCfit=[];
meanF1scorefit=[];stdF1scorefit=[];        
meanAUCfit=[];stdAUCfit=[];


# 病例的特征及其衍生变量共151个，对151种特征子集进行计算
for i in range(151):
    print('开始 ---------- 第%s个参数 '%(i+1))
    mean_BER_score,std_BER_score,  mean_TPR_score,std_TPR_score,  mean_TNR_score,std_TNR_score,  mean_ACC_score,std_ACC_score,  mean_MCC_score,std_MCC_score,  mean_F1score_score,std_F1score_score,  mean_AUC_score,std_AUC_score=RUN(jj=i+1)
    
    #存储不同特征子集下交叉验证结果的平均值与标准差
    meanBERfit.append(mean_BER_score);stdBERfit.append(std_BER_score);
    meanTPRfit.append(mean_TPR_score);stdTPRfit.append(std_TPR_score);
    meanTNRfit.append(mean_TNR_score);stdTNRfit.append(std_TNR_score);
    meanACCfit.append(mean_ACC_score);stdACCfit.append(std_ACC_score);
    meanMCCfit.append(mean_MCC_score);stdMCCfit.append(std_MCC_score);
    meanF1scorefit.append(mean_F1score_score);stdF1scorefit.append(std_F1score_score);
    meanAUCfit.append(mean_AUC_score);stdAUCfit.append(std_AUC_score);


meanBERfit = np.array(meanBERfit);meanTPRfit = np.array(meanTPRfit);meanTNRfit = np.array(meanTNRfit)
meanACCfit = np.array(meanACCfit);meanMCCfit = np.array(meanMCCfit);meanF1scorefit = np.array(meanF1scorefit)
meanAUCfit = np.array(meanAUCfit)
c={"meanBER":meanBERfit,"meanTPR":meanTPRfit,"meanTNR":meanTNRfit,
   "meanACC":meanACCfit,"meanMCC":meanMCCfit,"meanF1score":meanF1scorefit,
   "meanAUC":meanAUCfit}
writemean=pd.DataFrame(c)
writemean.to_csv('mean.csv', encoding='utf-8', index=True)   #将不同特征子集的平均值以CSV格式输出



stdBERfit = np.array(stdBERfit);stdTPRfit = np.array(stdTPRfit);stdTNRfit = np.array(stdTNRfit)
stdACCfit = np.array(stdACCfit);stdMCCfit = np.array(stdMCCfit);stdF1scorefit = np.array(stdF1scorefit)
stdAUCfit = np.array(stdAUCfit)
z={"stdBER":stdBERfit,"stdTPR":stdTPRfit,"stdTNR":stdTNRfit,
   "stdACC":stdACCfit,"stdMCC":stdMCCfit,"stdF1score":stdF1scorefit,
   "stdAUC":stdAUCfit}
writestd=pd.DataFrame(z)
writestd.to_csv('std.csv', encoding='utf-8', index=True)  #将不同特征子集的标准差以CSV格式输出
