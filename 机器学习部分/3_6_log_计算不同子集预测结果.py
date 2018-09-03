
# Description 
# 程序功能：计算不同特征子集下的预测结果的评价指标
# 程序流程：
#       Step1：对测试集部分进行预测
#       Step2：根据预测结果计算评价指标
# 程序运行结果：输出预测结果的评价指标
#
# DataFile: 病例为1*N的向量
#   0808_16g_3kinds_order.csv 进行特征排序后的数据集
#
# Output:
#   writemean    预测结果评价指标的平均值
#   writestd     预测结果评价指标的标准差
# V1.0 2018/8/28








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


alltime_start=time.time()


# 提取测试集，训练集
#comtest= pd.read_csv("new_0712_pat_with_characteristic_cat_allrunin_complete_in_rf_part_dim_reduction_onehot.csv")

comtest= pd.read_csv("0808_16g_3kinds_order.csv")

scaler = StandardScaler()  #数据标准化
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
    AUC = roc_auc_score(y_true,y_test_value[:,1])
    
    c={"TPR" : TPR,"TNR" : TNR,"BER" : BER
    ,"ACC" : ACC,"MCC" : MCC,"F1_score" : F1score,"AUC" : AUC}
    return c



def RUN(jj):
    comm_s_TPR=[];comm_s_TNR=[];comm_s_BER=[];comm_s_ACC=[];comm_s_MCC=[];comm_s_F1score=[];comm_s_AUC=[]; 
   
    # 提取测试集
    tiaocan_train, ceshi_train, tiaocan_train_test, ceshi_true = cross_validation.train_test_split(comtest.iloc[0:len(comtest),1:comtest.shape[1]-1],comtest.iloc[0:len(comtest),-1]
     , test_size = 0.2,random_state = random.randint(0,151)) 
    for times in range(10):
         # 提取验证集  训练集
        x_train, x_test, y_train, y_true = cross_validation.train_test_split(tiaocan_train, tiaocan_train_test
                                                                             , test_size = 0.1,random_state = times)
        x_train = np.c_[x_train, y_train]
#        x_train, y_train = RandomOverSampler().fit_sample(x_train, y_train)
        x_train, y_train = RandomUnderSampler().fit_sample(x_train, y_train)  #对训练集进行欠拟合
        x_train = x_train[:,0:x_train.shape[1]-1]
        ####  测试集  ##################################
        del x_test,y_true

        x_test = ceshi_train 
        y_true = ceshi_true
        ###############################################
        x_test=np.array(x_test,dtype=np.float16)
        y_true=np.array(y_true,dtype=np.float16)
        
        if x_train.shape[1] == 1:
            x_train = np.c_[x_train, x_train]
            x_test = np.c_[x_test, x_test]
        x_train = x_train[:,0:jj]  
        x_test = x_test[:,0:jj]    

        # 使用机器学习模型进行预测
        comm = LogisticRegression(solver='saga',warm_start=True)
        comm.fit(x_train , y_train)
        blo_comm_Pre = comm.predict(x_test)        
        pro_comm_Pre = comm.predict_proba(x_test)
           #对预测结果进行评价
        eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre)
        print('done_ 第%s次验证 '%(times+1))
        
        
        mean_BER_score=[];std_BER_score=[];
        mean_TPR_score=[];std_TPR_score=[];
        mean_TNR_score=[];std_TNR_score=[];
        mean_ACC_score=[];std_ACC_score=[];
        mean_MCC_score=[];std_MCC_score=[];
        mean_F1score_score=[];std_F1score_score=[];        
        mean_AUC_score=[];std_AUC_score=[];
        

        comm_s_BER.append(eva_comm['BER']);
        comm_s_TPR.append(eva_comm['TPR']);
        comm_s_TNR.append(eva_comm['TNR']);
        comm_s_ACC.append(eva_comm['ACC']);
        comm_s_MCC.append(eva_comm['MCC']);
        comm_s_F1score.append(eva_comm['F1_score']);
        comm_s_AUC.append(eva_comm['AUC']);
        
        #计算评价指标的平均值与标准差
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


for i in range(151):
    print('done_ ----------- 第%s个参数 '%(i+1))
    mean_BER_score,std_BER_score,  mean_TPR_score,std_TPR_score,  mean_TNR_score,std_TNR_score,  mean_ACC_score,std_ACC_score,  mean_MCC_score,std_MCC_score,  mean_F1score_score,std_F1score_score,  mean_AUC_score,std_AUC_score=RUN(jj=i+1)
    
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
writemean=pd.DataFrame(c)    #整理不同子集下的评价指标平均值
writemean.to_csv('1_mean.csv', encoding='utf-8', index=True)



stdBERfit = np.array(stdBERfit);stdTPRfit = np.array(stdTPRfit);stdTNRfit = np.array(stdTNRfit)
stdACCfit = np.array(stdACCfit);stdMCCfit = np.array(stdMCCfit);stdF1scorefit = np.array(stdF1scorefit)
stdAUCfit = np.array(stdAUCfit)
z={"stdBER":stdBERfit,"stdTPR":stdTPRfit,"stdTNR":stdTNRfit,
   "stdACC":stdACCfit,"stdMCC":stdMCCfit,"stdF1score":stdF1scorefit,
   "stdAUC":stdAUCfit}
writestd=pd.DataFrame(z)    #整理不同子集下的评价指标标准差
writestd.to_csv('1__std.csv', encoding='utf-8', index=True)
