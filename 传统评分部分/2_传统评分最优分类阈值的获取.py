# Description 
# 程序功能：根据训练集与验证集获取各个传统评分系统对应的最佳分类阈值
# 程序流程：
#       Step1：确定各个传统评分系统的阈值
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
from numpy import *
from sklearn.model_selection import StratifiedKFold


alltime_start=time.time()





# 提取测试集，训练集
#comtest= pd.read_csv("new_0712_pat_with_characteristic_cat_allrunin_complete_in_rf_part_dim_reduction_onehot.csv")

comtest= pd.read_csv("new_0731_with_traditional_score.csv")   #只包含传统评分与患者生死结果的数据，与 0808_16g_3kinds_order.csv 最后六列相同

#scaler = StandardScaler()
#comtest.iloc[:,0:comtest.shape[1]-1] = scaler.fit_transform(comtest.iloc[:,0:comtest.shape[1]-1])



def evaluating_indicator(y_true, y_test, y_test_value):   #计算模型预测结果的评价指标
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
    AUC = roc_auc_score(y_true,y_test_value)
    
    c={"TPR" : TPR,"TNR" : TNR,"BER" : BER
    ,"ACC" : ACC,"MCC" : MCC,"F1_score" : F1score,"AUC" : AUC}
    return c




def blo(pro_comm_Pre,jj):     # 根据预测病例的概率值与设定的分类阈值输出预测的布鲁值结果
    blo_Pre=zeros(int(len(pro_comm_Pre)))
    blo_Pre[(pro_comm_Pre>jj)]=1
    return blo_Pre



def RUN():   #主函数，选择最优分类阈值并计算相应阈值下验证集上预测结果的评价指标
    # 为每次交叉验证结果提供储存空间
    sapsii_s_TPR=[];sapsii_s_TNR=[];sapsii_s_BER=[];sapsii_s_ACC=[];sapsii_s_MCC=[];sapsii_s_F1score=[];sapsii_s_AUC=[];sapsii_s_th=[];   
    sofa_s_TPR=[];sofa_s_TNR=[];sofa_s_BER=[];sofa_s_ACC=[];sofa_s_MCC=[];sofa_s_F1score=[];sofa_s_AUC=[];sofa_s_th=[];
    apsiii_s_TPR=[];apsiii_s_TNR=[];apsiii_s_BER=[];apsiii_s_ACC=[];apsiii_s_MCC=[];apsiii_s_F1score=[];apsiii_s_AUC=[];apsiii_s_th=[];
    mews_s_TPR=[];mews_s_TNR=[];mews_s_BER=[];mews_s_ACC=[];mews_s_MCC=[];mews_s_F1score=[];mews_s_AUC=[];mews_s_th=[];
    oasis_s_TPR=[];oasis_s_TNR=[];oasis_s_BER=[];oasis_s_ACC=[];oasis_s_MCC=[];oasis_s_F1score=[];oasis_s_AUC=[];oasis_s_th=[];
    # 在固定随机种子的情况下将原始数据集分为 8:2 的（训练集+验证集）：测试集
    tiaocan_train, ceshi_train, tiaocan_train_test, ceshi_true = cross_validation.train_test_split(comtest.iloc[0:len(comtest),1:comtest.shape[1]-1],comtest.iloc[0:len(comtest),-1], test_size = 0.2,random_state = 0)    


    skf=StratifiedKFold(n_splits=10)    # 设置接下来的十折交叉验证
    tiaocan_train=np.array(tiaocan_train,dtype=np.float16)
    tiaocan_train_test=np.array(tiaocan_train_test,dtype=np.float16)
    times=0
    
    for train, test in skf.split(tiaocan_train,tiaocan_train_test): #进行十折交叉验证
        times=times+1

        #生成 训练集与验证集
        x_train_trad=tiaocan_train[train]
        y_train=tiaocan_train_test[train]
        x_test_trad=tiaocan_train[test]
        y_true=tiaocan_train_test[test]  

    
#        提取sapsii评分训练集，验证集数据
        sapsii_train =x_train_trad[:,0]
        sapsii_test =x_test_trad[:,0]
        
        position=[]
        RightIndex=[]
        for jj in range(int(max(sapsii_train))): #计算不同阈值下预测结果的各项指标
            blo_comm_Pre = blo(sapsii_train,jj)
            eva_comm = evaluating_indicator(y_true=y_train, y_test=blo_comm_Pre, y_test_value=sapsii_train)
            RightIndex.append(abs(eva_comm['TPR'] - eva_comm['TNR']))  
        RightIndex=np.array(RightIndex,dtype=np.float16)
        position=np.argmin(RightIndex)    #选择使敏感性与特异性差别最小的阈值
        th=position;sapsii_s_th.append(th); #储存每次交叉验证后的最优阈值
        blo_comm_Pre = blo(sapsii_test,th)  #计算预测结果的布鲁值
        print('done_10_sapsii, 第%s次验证 '%(times+1))
#        sapsii_test = np.c_[sapsii_test, sapsii_test]
        eva_sapsii = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value = sapsii_test)   
        # 储存每次交叉验证指标
        sapsii_s_TPR.append(eva_sapsii['TPR']);sapsii_s_TNR.append(eva_sapsii['TNR']);sapsii_s_BER.append(eva_sapsii['BER']);
        sapsii_s_ACC.append(eva_sapsii['ACC']);sapsii_s_MCC.append(eva_sapsii['MCC']);sapsii_s_F1score.append(eva_sapsii['F1_score']);
        sapsii_s_AUC.append(eva_sapsii['AUC']);
        
        
#        sofa   
        sofa_train =x_train_trad[:,1]
        sofa_test =x_test_trad[:,1]
        
        position=[]
        RightIndex=[]
        for jj in range(int(max(sofa_train))):#计算不同阈值下预测结果的各项指标
            blo_comm_Pre = blo( sofa_train,jj)
            eva_comm = evaluating_indicator(y_true=y_train, y_test=blo_comm_Pre, y_test_value=sofa_train)
            RightIndex.append(abs(eva_comm['TPR'] - eva_comm['TNR']))
        RightIndex=np.array(RightIndex,dtype=np.float16)
        position=np.argmin(RightIndex)   #选择使敏感性与特异性差别最小的阈值
        th=position;sofa_s_th.append(th);   #储存每次交叉验证后的最优阈值
        blo_comm_Pre = blo(sofa_test ,th)  #计算预测结果的布鲁值
        print('done_10_sofa, 第%s次验证 '%(times+1))
#        sofa_test = np.c_[sofa_test, sofa_test]
        eva_sofa = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value = sofa_test)      
        # 储存每次交叉验证指标
        sofa_s_TPR.append(eva_sofa['TPR']);sofa_s_TNR.append(eva_sofa['TNR']);sofa_s_BER.append(eva_sofa['BER']);
        sofa_s_ACC.append(eva_sofa['ACC']);sofa_s_MCC.append(eva_sofa['MCC']);sofa_s_F1score.append(eva_sofa['F1_score']);
        sofa_s_AUC.append(eva_sofa['AUC']);
        
        
#        apsiii
        apsiii_train =x_train_trad[:,2]
        apsiii_test =x_test_trad[:,2]
        
        position=[]
        RightIndex=[]
        for jj in range(int(max(apsiii_train))):  #计算不同阈值下预测结果的各项指标
            blo_comm_Pre = blo( apsiii_train,jj)
            eva_comm = evaluating_indicator(y_true=y_train, y_test=blo_comm_Pre, y_test_value=apsiii_train)
            RightIndex.append(abs(eva_comm['TPR'] - eva_comm['TNR']))
        RightIndex=np.array(RightIndex,dtype=np.float16)
        position=np.argmin(RightIndex)   #选择使敏感性与特异性差别最小的阈值
        th=position;apsiii_s_th.append(th);   #储存每次交叉验证后的最优阈值
        blo_comm_Pre = blo( apsiii_test ,th)  #计算预测结果的布鲁值
        print('done_10_apsiii, 第%s次验证 '%(times+1))
#        apsiii_test = np.c_[apsiii_test, apsiii_test]
        eva_apsiii = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value = apsiii_test) 
        # 储存每次交叉验证指标
        apsiii_s_TPR.append(eva_apsiii['TPR']);apsiii_s_TNR.append(eva_apsiii['TNR']);apsiii_s_BER.append(eva_apsiii['BER']);
        apsiii_s_ACC.append(eva_apsiii['ACC']);apsiii_s_MCC.append(eva_apsiii['MCC']);apsiii_s_F1score.append(eva_apsiii['F1_score']);
        apsiii_s_AUC.append(eva_apsiii['AUC']);
        
        
#        mews
        mews_train =x_train_trad[:,3]
        mews_test =x_test_trad[:,3]
        
        position=[]
        RightIndex=[]
        for jj in range(int(max(mews_train))): #计算不同阈值下预测结果的各项指标
            blo_comm_Pre = blo( mews_train,jj)
            eva_comm = evaluating_indicator(y_true=y_train, y_test=blo_comm_Pre, y_test_value=mews_train)
            RightIndex.append(abs(eva_comm['TPR'] - eva_comm['TNR']))
        RightIndex=np.array(RightIndex,dtype=np.float16)
        position=np.argmin(RightIndex)     #选择使敏感性与特异性差别最小的阈值
        th=position;mews_s_th.append(th);    #储存每次交叉验证后的最优阈值
        blo_comm_Pre = blo(mews_test,th)  #计算预测结果的布鲁值
        print('done_10_mews, 第%s次验证 '%(times+1))
#        mews_test = np.c_[mews_test, mews_test]
        eva_mews = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value = mews_test)  
        # 储存每次交叉验证指标
        mews_s_TPR.append(eva_mews['TPR']);mews_s_TNR.append(eva_mews['TNR']);mews_s_BER.append(eva_mews['BER']);
        mews_s_ACC.append(eva_mews['ACC']);mews_s_MCC.append(eva_mews['MCC']);mews_s_F1score.append(eva_mews['F1_score']);
        mews_s_AUC.append(eva_mews['AUC']);
        
        
#        oasis       
        oasis_train =x_train_trad[:,4]
        oasis_test =x_test_trad[:,4]
        
        position=[]
        RightIndex=[]
        for jj in range(int(max(oasis_train))): #计算不同阈值下预测结果的各项指标
            blo_comm_Pre = blo( oasis_train,jj)
            eva_comm = evaluating_indicator(y_true=y_train, y_test=blo_comm_Pre, y_test_value=oasis_train)
            RightIndex.append(abs(eva_comm['TPR'] - eva_comm['TNR']))
        RightIndex=np.array(RightIndex,dtype=np.float16)
        position=np.argmin(RightIndex)     #选择使敏感性与特异性差别最小的阈值
        th=position;oasis_s_th.append(th);      #储存每次交叉验证后的最优阈值
        blo_comm_Pre = blo(oasis_test,th)  #计算预测结果的布鲁值
        print('done_10_oasis, 第%s次验证 '%(times+1))
#        oasis_test = np.c_[oasis_test, oasis_test]
        eva_oasis = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value = oasis_test)  
        # 储存每次交叉验证指标
        oasis_s_TPR.append(eva_oasis['TPR']);oasis_s_TNR.append(eva_oasis['TNR']);oasis_s_BER.append(eva_oasis['BER']);
        oasis_s_ACC.append(eva_oasis['ACC']);oasis_s_MCC.append(eva_oasis['MCC']);oasis_s_F1score.append(eva_oasis['F1_score']);
        oasis_s_AUC.append(eva_oasis['AUC']);      
        
        

    #计算并保存交叉验证预测结果指标的平均值
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