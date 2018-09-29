
# Description 
# 程序功能：通过训练集与验证集获取最优分类阈值，并应用到不同子集的测试集上进行结果预测
# 程序流程：
#       Step1：通过训练集与验证集获取使得敏感性与特异性最为相近的分类阈值
#       Step2：将分类阈值使用到测试集预测中
# 程序运行结果：输出预测结果评价指标
#
# DataFile: 病例为1*N的向量
#   0808_16g_3kinds_order.csv 进行特征排序后的数据集
#
# Output:
# eva_comm_best 最优分类阈值下机器学习模型在测试集上的预测结果
# V1.0 2018/9/14

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

from sklearn.model_selection import StratifiedKFold




# 提取测试集，训练集
#comtest= pd.read_csv("new_0712_pat_with_characteristic_cat_allrunin_complete_in_rf_part_dim_reduction_onehot.csv")

comtest = pd.read_csv("0808_16g_3kinds_order.csv")
tt=151;   #提取数据的前 tt 个特征
z=[]
for i in range(tt+1):
    z.append(i)
z.append(comtest.shape[1]-1)
comtest = comtest.iloc[:,z]


scaler = StandardScaler()   #对病例数据进行标准化处理
comtest.iloc[:,0:comtest.shape[1]-1] = scaler.fit_transform(comtest.iloc[:,0:comtest.shape[1]-1])



def evaluating_indicator(y_true, y_test, y_test_value):  #计算预测结果的各项指标
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


def blo(pro_comm_Pre,jj):     #根据预测概率与最优分类阈值对患者进行生死预测
    blo_Pre=zeros(len(pro_comm_Pre))
    blo_Pre[(pro_comm_Pre[:,1]>(jj*0.01))]=1
    return blo_Pre


def RUN():   #根据训练集与验证集获取最优分类阈值
    tiaocan_train, ceshi_train, tiaocan_train_test, ceshi_true = cross_validation.train_test_split(comtest.iloc[0:len(comtest),1:comtest.shape[1]-1],comtest.iloc[0:len(comtest),-1], test_size = 0.2,random_state = 0)    
    position=[];
    skf=StratifiedKFold(n_splits=10)  #设置十折交叉验证
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
#        x_train, y_train = RandomUnderSampler().fit_sample(x_train, y_train)   #使用欠采样的方法进行类平衡
        
        # 设置机器学习模型
##########################################################################################################################
##########################################################################################################################
        
############################# --regr-- #############################
        comm = RandomForestClassifier(n_estimators=500, criterion='entropy', oob_score=True,min_samples_split=6,
                                      random_state=6,  min_samples_leaf=6,n_jobs=-1)
        
############################## --XGB-- #############################
#        comm = xgb.XGBClassifier(learning_rate =0.01, n_estimators=5000, max_depth=5, min_child_weight=5,n_jobs=8,
#                                tree_method='exact',objective = 'rank:pairwise',
#                                colsample_bytree=0.8, reg_alpha=0.005)
#        
############################## --GBM-- #############################
#        comm = lgb.LGBMClassifier(max_bin = 255,
#                                 num_leaves=1000,learning_rate =0.005,n_estimators=7000,n_jobs=-1,
#                                 reg_alpha=0.08,max_depth=10, min_child_weight=10)  
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
#        a=0.2
#        comm = GNB=GaussianNB(priors=[a,1-a])
#        
############################## --DT-- #############################
#        comm = DecisionTreeClassifier(splitter='random',min_samples_split=40,
#                                  min_samples_leaf=80 ,max_leaf_nodes=None)
 
##########################################################################################################################  
##########################################################################################################################
        comm.fit(x_train , y_train)    #对机器学习模型进行训练
        pro_comm_Pre = comm.predict_proba(x_test)


############################### 敏感性特异性相近 ########################################
        RightIndex=[]
        for jj in range(100): #计算模型在不同分类阈值下的各项指标
            blo_comm_Pre = blo(pro_comm_Pre,jj)
            eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre)
            RightIndex.append(abs(eva_comm['TPR'] - eva_comm['TNR']))
        RightIndex=np.array(RightIndex,dtype=np.float16)
        position=np.argmin(RightIndex)  #选择出使得敏感性特异性最小的阈值作为分类阈值输出
        alltime_end=time.time()
        print('done_0, 第%s次验证   , time: %s  s '%(times  alltime_end-alltime_start)) 
######################################################################################
    return  position.mean()  #计算交叉验证输出的多个阈值的平均值作为最优分类阈值
best_th = RUN()






print(' done_1  best_th')

def RUN_2(best_th):   #主函数，在获得最优分类阈值的情况下计算模型在测试集上的预测结果
    comm_s_TPR=[];comm_s_TNR=[];comm_s_BER=[];comm_s_ACC=[];comm_s_MCC=[];comm_s_F1score=[];comm_s_AUC=[];comm_s_time=[];
    #将原始数据分为训练集，测试集
    tiaocan_train, ceshi_train, tiaocan_train_test, ceshi_true = cross_validation.train_test_split(comtest.iloc[0:len(comtest),1:comtest.shape[1]-1],comtest.iloc[0:len(comtest),-1], test_size = 0.2,random_state = 0)    


    x_train = tiaocan_train
    y_train = tiaocan_train_test
    x_test = ceshi_train
    y_true = ceshi_true
    
    
    x_train=np.array(x_train,dtype=np.float16)
    y_train=np.array(y_train,dtype=np.float16)
    x_test=np.array(x_test,dtype=np.float16)
    y_true=np.array(y_true,dtype=np.float16)
#    x_train, y_train = RandomUnderSampler().fit_sample(x_train, y_train)  #对训练集使用欠采样的方法达到类平衡
    
        # 设置机器学习模型
##########################################################################################################################
##########################################################################################################################
        
############################# --regr-- #############################
    comm = RandomForestClassifier(n_estimators=500, criterion='entropy', oob_score=True,min_samples_split=6,
                                  random_state=6,  min_samples_leaf=6,n_jobs=-1)
        
############################## --XGB-- #############################
#    comm = xgb.XGBClassifier(learning_rate =0.01, n_estimators=5000, max_depth=5, min_child_weight=5,n_jobs=8,
#                            tree_method='exact',objective = 'rank:pairwise',
#                            colsample_bytree=0.8, reg_alpha=0.005)
#    
############################## --GBM-- #############################
#    comm = lgb.LGBMClassifier(max_bin = 255,
#                             num_leaves=1000,learning_rate =0.005,n_estimators=7000,n_jobs=-1,
#                             reg_alpha=0.08,max_depth=10, min_child_weight=10)  
#    
############################## --Adb-- #############################
#    comm = ensemble.AdaBoostClassifier(learning_rate =0.1, n_estimators=500)
#    
############################## --regr-- #############################
#    comm = svm.SVC(C=0.999, probability=True) 
#    
############################## --Log-- #############################
#    comm = LogisticRegression(solver='saga',warm_start=True)
#    
############################## --GBN-- #############################
#    a=0.2
#    comm = GNB=GaussianNB(priors=[a,1-a])
#    
############################## --DT-- #############################
#    comm = DecisionTreeClassifier(splitter='random',min_samples_split=40,
#                              min_samples_leaf=80 ,max_leaf_nodes=None)
# 
##########################################################################################################################  
##########################################################################################################################
    comm.fit(x_train , y_train)  #模型训练
    pro_comm_Pre = comm.predict_proba(x_test)
    blo_comm_Pre = blo(pro_comm_Pre,best_th)  #根据最优分类阈值与预测概率计算画着生死情况
    eva_comm = evaluating_indicator(y_true=y_true, y_test=blo_comm_Pre, y_test_value=pro_comm_Pre)
 
    comm_s_TPR.append(eva_comm['TPR']);comm_s_TNR.append(eva_comm['TNR']);comm_s_BER.append(eva_comm['BER']);
    comm_s_ACC.append(eva_comm['ACC']);comm_s_MCC.append(eva_comm['MCC']);comm_s_F1score.append(eva_comm['F1_score']);
    comm_s_AUC.append(eva_comm['AUC']);
    eva_comm={"TPR" : np.mean(comm_s_TPR),"TNR" : np.mean(comm_s_TNR),"BER" :  np.mean(comm_s_BER)
    ,"ACC" : np.mean(comm_s_ACC),"MCC" : np.mean(comm_s_MCC),"F1_score" : np.mean(comm_s_F1score)
    ,"AUC" : np.mean(comm_s_AUC),"time" : np.mean(comm_s_time)}    
    
    return  eva_comm
time_start=time.time()
eva_comm_best = RUN_2(best_th)
time_end=time.time()
print(' done_2  eva_comm_best,  time: %s s '  %(time_end-time_start))

time_start=time.time()
eva_comm_50 = RUN_2(50) #计算患者在默认阈值情况下预测结果的各项指标
time_end=time.time()
print(' done_3  eva_comm_50 ,  time: %s s '  %(time_end-time_start))


