# Description 
# 程序功能：计算根据传统评分系统所预测出的患者生死预测评价结果
# 程序流程：
#       Step1：确定各个传统评分系统的阈值
#       Step2：计算预测结果的评价指标数值
# 程序运行结果：各个传统评分系统预测结果的评价指标数值
#
# DataFile: 病例为1*N的向量
#   0808_16g_3kinds_order.csv   包含传统评分模型数值的数据集
#
# Output:
#    eva_sapsii,eva_sofa,eva_apsiii,eva_mews,eva_oasis   各个传统评分系统的评价结果
# V1.0 2018/8/28


import pandas as pd
import time
from sklearn import cross_validation
import numpy as np
import random


#评价指标
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef # MCC
from sklearn.metrics import confusion_matrix #混淆矩阵
from sklearn.preprocessing import StandardScaler

# 传统评分用
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import scipy.io as sio  
from sklearn.metrics import confusion_matrix  
from sklearn import metrics
import itertools
import copy


alltime_start=time.time()





# 提取测试集，训练集
#comtest= pd.read_csv("new_0712_pat_with_characteristic_cat_allrunin_complete_in_rf_part_dim_reduction_onehot.csv")

#comtest= pd.read_csv("0808_16g_3kinds_order.csv")
#scaler = StandardScaler()
#comtest.iloc[:,0:comtest.shape[1]-1] = scaler.fit_transform(comtest.iloc[:,0:comtest.shape[1]-1])

comtest= pd.read_csv("0808_16g_3kinds_order.csv")
#scaler = StandardScaler()
#comtest.iloc[:,0:comtest.shape[1]-6] = scaler.fit_transform(comtest.iloc[:,0:comtest.shape[1]-6])

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
    AUC = roc_auc_score(y_true,y_test_value[:,1])
    
    c={"TPR" : TPR,"TNR" : TNR,"BER" : BER
    ,"ACC" : ACC,"MCC" : MCC,"F1_score" : F1score,"AUC" : AUC}
    return c





def RUN():
    
    #创建传统评分评价指标数值存放位置
    sapsii_s_TPR=[];sapsii_s_TNR=[];sapsii_s_BER=[];sapsii_s_ACC=[];sapsii_s_MCC=[];sapsii_s_F1score=[];sapsii_s_AUC=[];sapsii_s_th=[];
    sofa_s_TPR=[];sofa_s_TNR=[];sofa_s_BER=[];sofa_s_ACC=[];sofa_s_MCC=[];sofa_s_F1score=[];sofa_s_AUC=[];sofa_s_th=[];
    apsiii_s_TPR=[];apsiii_s_TNR=[];apsiii_s_BER=[];apsiii_s_ACC=[];apsiii_s_MCC=[];apsiii_s_F1score=[];apsiii_s_AUC=[];apsiii_s_th=[];
    mews_s_TPR=[];mews_s_TNR=[];mews_s_BER=[];mews_s_ACC=[];mews_s_MCC=[];mews_s_F1score=[];mews_s_AUC=[];mews_s_th=[];
    oasis_s_TPR=[];oasis_s_TNR=[];oasis_s_BER=[];oasis_s_ACC=[];oasis_s_MCC=[];oasis_s_F1score=[];oasis_s_AUC=[];oasis_s_th=[];
    
    #划分出测试集
    tiaocan_train, ceshi_train, tiaocan_train_test, ceshi_true = cross_validation.train_test_split(comtest.iloc[0:len(comtest),1:comtest.shape[1]-1],comtest.iloc[0:len(comtest),-1], test_size = 0.2,random_state = random.randint(0,9))    
    
    
    for times in range(10):

            #划分出验证集，训练集
        x_train, x_test, y_train, y_true = cross_validation.train_test_split(tiaocan_train, tiaocan_train_test
                                                                             , test_size = 0.1,random_state = times)
        
        x_train = np.c_[x_train, y_train]
#        x_train, y_train = RandomOverSampler().fit_sample(x_train, y_train)
#        x_train, y_train = RandomUnderSampler().fit_sample(x_train, y_train)
        x_train = x_train[:,0:x_train.shape[1]-1]
       
#        x_train=np.array(x_train,dtype=np.float16)
#        y_train=np.array(y_train,dtype=np.float16)
        
        ####  测试集  ##################################  使用测试集对结果进行预测
        del x_test,y_true

        x_test = ceshi_train 
        y_true = ceshi_true
        ###############################################
        x_test=np.array(x_test,dtype=np.float16)
        y_true=np.array(y_true,dtype=np.float16)
       
            
        x_train_trad = x_train[:,x_train.shape[1]-5:x_train.shape[1]]   #分离出传统评分系统数值
        x_test_trad = x_test[:,x_test.shape[1]-5:x_test.shape[1]]


        
#        sapsii
        sapsii_train =x_train_trad[:,0]
        sapsii_test =x_test_trad[:,0]
        
        blo_sapsii_Pre=[]
        fpr, tpr, thresholds = roc_curve(y_train, sapsii_train, pos_label=1)   #计算每个数值对应的横纵坐标
        RightIndex=(tpr+(1-fpr)-1) #根据ROC横纵坐标计算一个与MCC同分布的数据集
        positon=np.argmax(RightIndex)  #选择使RI最大的数值为阈值
        aw=int(positon)   
        th=thresholds[aw];sapsii_s_th.append(th); 
        for i in range(len(sapsii_test)):   #对测试集进行预测
            if sapsii_test[i]>=th:
                blo_sapsii_Pre.append(1)
            else:
                blo_sapsii_Pre.append(0)
        print('done_10_sapsii, 第%s次验证 '%(times+1))
        sapsii_test = np.c_[sapsii_test, sapsii_test]
        eva_sapsii = evaluating_indicator(y_true=y_true, y_test=blo_sapsii_Pre, y_test_value = sapsii_test)      
        sapsii_s_TPR.append(eva_sapsii['TPR']);sapsii_s_TNR.append(eva_sapsii['TNR']);sapsii_s_BER.append(eva_sapsii['BER']);
        sapsii_s_ACC.append(eva_sapsii['ACC']);sapsii_s_MCC.append(eva_sapsii['MCC']);sapsii_s_F1score.append(eva_sapsii['F1_score']);
        sapsii_s_AUC.append(eva_sapsii['AUC']);
        
        
#        sofa
        sofa_train =x_train_trad[:,1]
        sofa_test =x_test_trad[:,1]
        
        blo_sofa_Pre=[]
        fpr, tpr, thresholds = roc_curve(y_train, sofa_train, pos_label=1)
        RightIndex=(tpr+(1-fpr)-1) #根据ROC横纵坐标计算一个与MCC同分布的数据集
        positon=np.argmax(RightIndex)  #选择使RI最大的数值为阈值
        aw=int(positon)   
        th=thresholds[aw];sapsii_s_th.append(th); 
        for i in range(len(sapsii_test)):   #对测试集进行预测
            if sofa_test[i]>=th:
                blo_sofa_Pre.append(1)
            else:
                blo_sofa_Pre.append(0)
        print('done_11_sofa, 第%s次验证 '%(times+1))
        sofa_test = np.c_[sofa_test, sofa_test]
        eva_sofa = evaluating_indicator(y_true=y_true, y_test=blo_sofa_Pre, y_test_value = sofa_test)      
        sofa_s_TPR.append(eva_sofa['TPR']);sofa_s_TNR.append(eva_sofa['TNR']);sofa_s_BER.append(eva_sofa['BER']);
        sofa_s_ACC.append(eva_sofa['ACC']);sofa_s_MCC.append(eva_sofa['MCC']);sofa_s_F1score.append(eva_sofa['F1_score']);
        sofa_s_AUC.append(eva_sofa['AUC']);
        
        
#        apsiii
        apsiii_train =x_train_trad[:,2]
        apsiii_test =x_test_trad[:,2]
        
        blo_apsiii_Pre=[]
        fpr, tpr, thresholds = roc_curve(y_train, apsiii_train, pos_label=1)
        RightIndex=(tpr+(1-fpr)-1) #根据ROC横纵坐标计算一个与MCC同分布的数据集
        positon=np.argmax(RightIndex)  #选择使RI最大的数值为阈值
        aw=int(positon)   
        th=thresholds[aw];sapsii_s_th.append(th); 
        for i in range(len(sapsii_test)):   #对测试集进行预测
            if apsiii_test[i]>=th:
                blo_apsiii_Pre.append(1)
            else:
                blo_apsiii_Pre.append(0)
        print('done_12_apsiii, 第%s次验证 '%(times+1))
        apsiii_test = np.c_[apsiii_test, apsiii_test]
        eva_apsiii = evaluating_indicator(y_true=y_true, y_test=blo_apsiii_Pre, y_test_value = apsiii_test)      
        apsiii_s_TPR.append(eva_apsiii['TPR']);apsiii_s_TNR.append(eva_apsiii['TNR']);apsiii_s_BER.append(eva_apsiii['BER']);
        apsiii_s_ACC.append(eva_apsiii['ACC']);apsiii_s_MCC.append(eva_apsiii['MCC']);apsiii_s_F1score.append(eva_apsiii['F1_score']);
        apsiii_s_AUC.append(eva_apsiii['AUC']);
        
        
#        mews
        mews_train =x_train_trad[:,3]
        mews_test =x_test_trad[:,3]
        
        blo_mews_Pre=[]
        fpr, tpr, thresholds = roc_curve(y_train, mews_train, pos_label=1)
        RightIndex=(tpr+(1-fpr)-1) #根据ROC横纵坐标计算一个与MCC同分布的数据集
        positon=np.argmax(RightIndex)  #选择使RI最大的数值为阈值
        aw=int(positon)   
        th=thresholds[aw];sapsii_s_th.append(th); 
        for i in range(len(sapsii_test)):   #对测试集进行预测
            if mews_test[i]>=th:
                blo_mews_Pre.append(1)
            else:
                blo_mews_Pre.append(0)
        print('done_13_mews, 第%s次验证 '%(times+1))
        mews_test = np.c_[mews_test, mews_test]
        eva_mews = evaluating_indicator(y_true=y_true, y_test=blo_mews_Pre, y_test_value = mews_test)      
        mews_s_TPR.append(eva_mews['TPR']);mews_s_TNR.append(eva_mews['TNR']);mews_s_BER.append(eva_mews['BER']);
        mews_s_ACC.append(eva_mews['ACC']);mews_s_MCC.append(eva_mews['MCC']);mews_s_F1score.append(eva_mews['F1_score']);
        mews_s_AUC.append(eva_mews['AUC']);
        
        
#        oasis       
        oasis_train =x_train_trad[:,4]
        oasis_test =x_test_trad[:,4]
        
        blo_oasis_Pre=[]
        fpr, tpr, thresholds = roc_curve(y_train, oasis_train, pos_label=1)
        RightIndex=(tpr+(1-fpr)-1) #根据ROC横纵坐标计算一个与MCC同分布的数据集
        positon=np.argmax(RightIndex)  #选择使RI最大的数值为阈值
        aw=int(positon)   
        th=thresholds[aw];sapsii_s_th.append(th); 
        for i in range(len(sapsii_test)):   #对测试集进行预测
            if oasis_test[i]>=th:
                blo_oasis_Pre.append(1)
            else:
                blo_oasis_Pre.append(0)
        print('done_14_oasis, 第%s次验证 '%(times+1))
        oasis_test = np.c_[oasis_test, oasis_test]
        eva_oasis = evaluating_indicator(y_true=y_true, y_test=blo_oasis_Pre, y_test_value = oasis_test)      
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

eva_sapsii,eva_sofa,eva_apsiii,eva_mews,eva_oasis=RUN()   # 输出对预测结果的评价

##########
alltime_end=time.time()

allruntime=alltime_start-alltime_end

del alltime_end
del alltime_start