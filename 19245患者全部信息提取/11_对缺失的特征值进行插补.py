# Description 
# 程序功能：使用随机森林对缺失数据进行插补
# 程序流程：
#       Step1：对数值为标量的变量进行独热编码
#       Step2：训练随机森林模型并对缺失值进行预测
# 程序运行结果：输出对缺失据插补后的数据集
#
# DataFile: 病例为1*N的向量
#   new_0710_pat_with_characteristic_cat_partrunin.csv   对BMI,TEMP,FIO2插补后的数据集
#
# Output:
#    tempData  对缺失数据数据进行插补后的数据集
# V1.0 2018/8/28


import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import xgboost as xgb
from inspect import getframeinfo

comtest= pd.read_csv("new_0710_pat_with_characteristic_cat_partrunin.csv")

# function
def nbpRandom(comtest, z):
    # abpNotNull = comtest[comtest[abp].notnull()]  # 后续只使用这一部分数据
    abpNotNull = comtest
    # 把abpNotNull分为nbpNull和nbpNotNull两部分
    nbpNull = abpNotNull[abpNotNull.ix[:,z].isnull()]  # 测试集
    nbpNotNull = abpNotNull[abpNotNull.ix[:,z].notnull()]  # 训练集
    
    #linshix = OneHotEncoder(sparse = False).fit_transform( nbpNotNull[['gender','death','vent','bmi_label']])#提取数据补齐所需要的变量并进行独热编码
    #X = np.hstack((linshix , nbpNotNull.loc[:, ['age',abp]]))
    linshi=list(range(1,154))
    
    #delect=[2,3,4,7,10,z]
    delect=[3,z]          # 整合需要进行插补的变量位置
    for i in delect:
        linshi.remove(i) 

    #X = np.hstack((linshix , nbpNotNull.ix[:, linshi]))
    X = nbpNotNull.ix[:, linshi]
    X = pd.DataFrame(X)
    t = X.dropna()
    X = X.fillna(t.mean())    
    
    y = nbpNotNull.ix[:,z]
    regr = RandomForestRegressor(random_state=10,min_samples_leaf=120,n_jobs=-1,warm_start=True)   
    regr.fit(X , y)   # 训练随机森林模型
    #linshiy = OneHotEncoder(sparse = False).fit_transform( nbpNull[['gender','death','vent','bmi_label']])
    #X_pre = np.hstack((linshiy , nbpNull.loc[:, ['age',abp]]))
    
    t = nbpNull.dropna()
    if len(t)==0:
        t = abpNotNull.dropna()
    nbpNull = nbpNull.fillna(t.mean()) 
    #X_pre = np.hstack((linshiy , nbpNull.ix[:, linshi]))
    X_pre = nbpNull.ix[:, linshi]
    
    nbpPre = regr.predict(X_pre)
    abpNotNull.ix[abpNotNull.ix[:,z].isnull(), z] = nbpPre
    comtest.ix[comtest.ix[:,z].isnull(), z] = abpNotNull.ix[:,z]
    return comtest

print('done1')

# 需要进行插补的变量位置
tt=[5,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,32,33,41,42,50,51,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,113,114,131,132,149,150]
for i in tt:
    comtest = nbpRandom(comtest, z=i)
    print ('当前填补第: %d 列' %(i))



z=5
