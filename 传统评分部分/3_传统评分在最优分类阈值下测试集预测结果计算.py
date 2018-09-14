# Description 
# 程序功能：在获取最佳分类阈值的基础上在测试集上进行结果预测
# 程序流程：
#       Step1：根据分类阈值对患者进行生死预测
#       Step2：计算预测结果的评价指标数值
# 程序运行结果：各个传统评分系统预测结果的评价指标数值
#
# DataFile: 数据为N*1的多个向量
#   0new_0731_with_traditional_score.csv   只包含传统评分模型数值的数据集
#
# Output:
#    eva_sapsii,eva_sofa,eva_apsiii,eva_mews,eva_oasis   各个传统评分系统的评价结果
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


alltime_start=time.time()




comtest= pd.read_csv("new_0731_with_traditional_score.csv")  #只包含传统评分与患者生死结果的数据，与 0808_16g_3kinds_order.csv 最后六列相同


def evaluating_indicator(y_true, y_test, y_test_value):    #计算模型预测结果的评价指标
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



def RUN():   #主函数，选择最优分类阈值并计算相应阈值下验证集上预测结果的评价指标
    # 为每次交叉验证结果提供储存空间
    sapsii_s_TPR=[];sapsii_s_TNR=[];sapsii_s_BER=[];sapsii_s_ACC=[];sapsii_s_MCC=[];sapsii_s_F1score=[];sapsii_s_AUC=[];sapsii_s_th=[];
    sofa_s_TPR=[];sofa_s_TNR=[];sofa_s_BER=[];sofa_s_ACC=[];sofa_s_MCC=[];sofa_s_F1score=[];sofa_s_AUC=[];sofa_s_th=[];
    apsiii_s_TPR=[];apsiii_s_TNR=[];apsiii_s_BER=[];apsiii_s_ACC=[];apsiii_s_MCC=[];apsiii_s_F1score=[];apsiii_s_AUC=[];apsiii_s_th=[];
    mews_s_TPR=[];mews_s_TNR=[];mews_s_BER=[];mews_s_ACC=[];mews_s_MCC=[];mews_s_F1score=[];mews_s_AUC=[];mews_s_th=[];
    oasis_s_TPR=[];oasis_s_TNR=[];oasis_s_BER=[];oasis_s_ACC=[];oasis_s_MCC=[];oasis_s_F1score=[];oasis_s_AUC=[];oasis_s_th=[];
    
    #将原始数据分为训练集与测试集
    tiaocan_train, ceshi_train, tiaocan_train_test, ceshi_true = cross_validation.train_test_split(comtest.iloc[0:len(comtest),1:comtest.shape[1]-1],comtest.iloc[0:len(comtest),-1], test_size = 0.2,random_state = 0)    


    times=0
    
    x_train = tiaocan_train
    y_train = tiaocan_train_test
    x_test = ceshi_train
    y_true = ceshi_true
    
    
    x_train_trad=np.array(x_train,dtype=np.float16)
    y_train=np.array(y_train,dtype=np.float16)
    x_test_trad=np.array(x_test,dtype=np.float16)
    y_true=np.array(y_true,dtype=np.float16)



#     提取sapsii评分训练集，测试集数据
    sapsii_train =x_train_trad[:,0]
    sapsii_test =x_test_trad[:,0]
    
    blo_sapsii_Pre=[]

    th= 40   #设置由上一步计算得到的最优分类阈值
    for i in range(len(sapsii_test)):  #根据分类阈值与传统评分结果对患者生死进行预测
        if sapsii_test[i]>=th:
            blo_sapsii_Pre.append(1)
        else:
            blo_sapsii_Pre.append(0)
    print('done_10_sapsii, 第%s次验证 '%(times+1))
    sapsii_test = np.c_[sapsii_test, sapsii_test]
    eva_sapsii = evaluating_indicator(y_true=y_true, y_test=blo_sapsii_Pre, y_test_value = sapsii_test)   
    # 储存验证结果的各项评价
    sapsii_s_TPR.append(eva_sapsii['TPR']);sapsii_s_TNR.append(eva_sapsii['TNR']);sapsii_s_BER.append(eva_sapsii['BER']);
    sapsii_s_ACC.append(eva_sapsii['ACC']);sapsii_s_MCC.append(eva_sapsii['MCC']);sapsii_s_F1score.append(eva_sapsii['F1_score']);
    sapsii_s_AUC.append(eva_sapsii['AUC']);
    
    
#        sofa
    sofa_train =x_train_trad[:,1]
    sofa_test =x_test_trad[:,1]
    
    blo_sofa_Pre=[]

    th= 5  #设置由上一步计算得到的最优分类阈值
    for i in range(len(sofa_test)):  #根据分类阈值与传统评分结果对患者生死进行预测
        if sofa_test[i]>=th:
            blo_sofa_Pre.append(1)
        else:
            blo_sofa_Pre.append(0)
    print('done_11_sofa, 第%s次验证 '%(times+1))
    sofa_test = np.c_[sofa_test, sofa_test]
    eva_sofa = evaluating_indicator(y_true=y_true, y_test=blo_sofa_Pre, y_test_value = sofa_test)      
    # 储存验证结果的各项评价
    sofa_s_TPR.append(eva_sofa['TPR']);sofa_s_TNR.append(eva_sofa['TNR']);sofa_s_BER.append(eva_sofa['BER']);
    sofa_s_ACC.append(eva_sofa['ACC']);sofa_s_MCC.append(eva_sofa['MCC']);sofa_s_F1score.append(eva_sofa['F1_score']);
    sofa_s_AUC.append(eva_sofa['AUC']);
    
    
#        apsiii
    apsiii_train =x_train_trad[:,2]
    apsiii_test =x_test_trad[:,2]
    
    blo_apsiii_Pre=[]
  
    th= 45.8    #设置由上一步计算得到的最优分类阈值
    for i in range(len(apsiii_test)):  #根据分类阈值与传统评分结果对患者生死进行预测
        if apsiii_test[i]>=th:
            blo_apsiii_Pre.append(1)
        else:
            blo_apsiii_Pre.append(0)
    print('done_12_apsiii, 第%s次验证 '%(times+1))
    apsiii_test = np.c_[apsiii_test, apsiii_test]
    eva_apsiii = evaluating_indicator(y_true=y_true, y_test=blo_apsiii_Pre, y_test_value = apsiii_test) 
    # 储存验证结果的各项评价     
    apsiii_s_TPR.append(eva_apsiii['TPR']);apsiii_s_TNR.append(eva_apsiii['TNR']);apsiii_s_BER.append(eva_apsiii['BER']);
    apsiii_s_ACC.append(eva_apsiii['ACC']);apsiii_s_MCC.append(eva_apsiii['MCC']);apsiii_s_F1score.append(eva_apsiii['F1_score']);
    apsiii_s_AUC.append(eva_apsiii['AUC']);
    
    
#        mews
    mews_train =x_train_trad[:,3]
    mews_test =x_test_trad[:,3]
    
    blo_mews_Pre=[]

    th= 5   #设置由上一步计算得到的最优分类阈值
    for i in range(len(mews_test)):  #根据分类阈值与传统评分结果对患者生死进行预测
        if mews_test[i]>=th:
            blo_mews_Pre.append(1)
        else:
            blo_mews_Pre.append(0)
    print('done_13_mews, 第%s次验证 '%(times+1))
    mews_test = np.c_[mews_test, mews_test]
    eva_mews = evaluating_indicator(y_true=y_true, y_test=blo_mews_Pre, y_test_value = mews_test)  
    # 储存验证结果的各项评价      
    mews_s_TPR.append(eva_mews['TPR']);mews_s_TNR.append(eva_mews['TNR']);mews_s_BER.append(eva_mews['BER']);
    mews_s_ACC.append(eva_mews['ACC']);mews_s_MCC.append(eva_mews['MCC']);mews_s_F1score.append(eva_mews['F1_score']);
    mews_s_AUC.append(eva_mews['AUC']);
    
    
#        oasis       
    oasis_train =x_train_trad[:,4]
    oasis_test =x_test_trad[:,4]
    
    blo_oasis_Pre=[]
 
    th= 36.5  #设置由上一步计算得到的最优分类阈值
    for i in range(len(oasis_test)):  #根据分类阈值与传统评分结果对患者生死进行预测
        if oasis_test[i]>=th:
            blo_oasis_Pre.append(1)
        else:
            blo_oasis_Pre.append(0)
    print('done_14_oasis, 第%s次验证 '%(times+1))
    oasis_test = np.c_[oasis_test, oasis_test]
    eva_oasis = evaluating_indicator(y_true=y_true, y_test=blo_oasis_Pre, y_test_value = oasis_test)  
    # 储存验证结果的各项评价     
    oasis_s_TPR.append(eva_oasis['TPR']);oasis_s_TNR.append(eva_oasis['TNR']);oasis_s_BER.append(eva_oasis['BER']);
    oasis_s_ACC.append(eva_oasis['ACC']);oasis_s_MCC.append(eva_oasis['MCC']);oasis_s_F1score.append(eva_oasis['F1_score']);
    oasis_s_AUC.append(eva_oasis['AUC']);        
        
        

    
    eva_sapsii={"TPR" : np.mean(sapsii_s_TPR),"TNR" : np.mean(sapsii_s_TNR),"BER" :  np.mean(sapsii_s_BER)
    ,"ACC" : np.mean(sapsii_s_ACC),"MCC" : np.mean(sapsii_s_MCC),"F1_score" : np.mean(sapsii_s_F1score)
    ,"AUC" : np.mean(sapsii_s_AUC),"th" : np.mean(sapsii_s_th)}
    
    eva_sofa={"TPR" : np.mean(sofa_s_TPR),"TNR" : np.mean(sofa_s_TNR),"BER" :  np.mean(sofa_s_BER)
    ,"ACC" : np.mean(sofa_s_ACC),"MCC" : np.mean(sofa_s_MCC),"F1_score" : np.mean(sofa_s_F1score)
    ,"AUC" : np.mean(sofa_s_AUC),"th" : np.mean(sofa_s_th)}
    
    eva_apsiii={"TPR" : np.mean(apsiii_s_TPR),"TNR" : np.mean(apsiii_s_TNR),"BER" :  np.mean(apsiii_s_BER)
    ,"ACC" : np.mean(apsiii_s_ACC),"MCC" : np.mean(apsiii_s_MCC),"F1_score" : np.mean(apsiii_s_F1score)
    ,"AUC" : np.mean(apsiii_s_AUC),"th" : np.mean(apsiii_s_th)}  
        
    eva_mews={"TPR" : np.mean(mews_s_TPR),"TNR" : np.mean(mews_s_TNR),"BER" :  np.mean(mews_s_BER)
    ,"ACC" : np.mean(mews_s_ACC),"MCC" : np.mean(mews_s_MCC),"F1_score" : np.mean(mews_s_F1score)
    ,"AUC" : np.mean(mews_s_AUC),"th" : np.mean(mews_s_th)}     
    
    eva_oasis={"TPR" : np.mean(oasis_s_TPR),"TNR" : np.mean(oasis_s_TNR),"BER" :  np.mean(oasis_s_BER)
    ,"ACC" : np.mean(oasis_s_ACC),"MCC" : np.mean(oasis_s_MCC),"F1_score" : np.mean(oasis_s_F1score)
    ,"AUC" : np.mean(oasis_s_AUC),"th" : np.mean(oasis_s_th)}   
    
    
    return eva_sapsii,eva_sofa,eva_apsiii,eva_mews,eva_oasis


eva_sapsii,eva_sofa,eva_apsiii,eva_mews,eva_oasis=RUN()


##########
alltime_end=time.time()

allruntime=alltime_start-alltime_end

del alltime_end
del alltime_start