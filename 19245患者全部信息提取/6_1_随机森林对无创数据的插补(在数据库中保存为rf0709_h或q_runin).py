# Description 
# 程序功能：使用随机森林对缺失的无创血压数据进行插补，使用参数有gender','icutype','age','vent'
# 程序流程：
#       Step1：对数值为标量的变量进行独热编码
#       Step2：训练随机森林模型并对缺失值进行预测
# 程序运行结果：输出对缺失的无创数据插补后的数据集
#
# DataFile: 病例为1*N的向量
#   sofa.csv   对无创数据进行插补前的数据集
#
# Output:
#    tempData  对无创数据进行插补后的数据集
# V1.0 2018/8/28



import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv("sofa.csv")    #此处数据为仅仅对无创血压数据补齐后的数据集

tempData = data.loc[:, ['subject_id', 'hadm_id', 'icustay_id'
                        ,'age','bmi','gender','icutype','death','vent','sysbp','diasbp','meanbp'
                        ,'nisysbp','nidiasbp','nimeanbp']]   #提取数据补齐所需要的变量


print('done1')



# function
def nbpRandom(tempData, nbp, abp):
    abpNotNull = tempData[tempData[abp].notnull()]  # 后续只使用这一部分数据，根据已知结果的数据对数据补齐模型进行调参
    # 把abpNotNull分为nbpNull和nbpNotNull两部分
    nbpNull = abpNotNull[abpNotNull[nbp].isnull()]  # 测试集
    nbpNotNull = abpNotNull[abpNotNull[nbp].notnull()]  # 训练集
    
    linshix = OneHotEncoder(sparse = False).fit_transform( nbpNotNull[['gender','icutype','vent']])  # 删除模型学习的训练集中的含有空值的患者信息
 

    X = np.hstack((linshix , nbpNotNull.loc[:, ['age',abp]]))
    
    #X = X.fillna(X.mean())
    y = nbpNotNull[nbp]
    regr = RandomForestRegressor(max_depth=2, random_state=0)
    regr.fit(X, y)
  
    linshiy = OneHotEncoder(sparse = False).fit_transform( nbpNull[['gender','icutype','death','vent']])   #对标量参变量进行独热编码，并整理数据补齐所需的参变量
    X_pre = np.hstack((linshiy , nbpNull.loc[:, ['age',abp]]))
    
    #X_pre = X_pre.fillna(X_pre.mean())
    nbpPre = regr.predict(X_pre)
    abpNotNull.loc[abpNotNull[nbp].isnull(), nbp] = nbpPre    
    tempData.loc[tempData[abp].notnull(), nbp] = abpNotNull[nbp]     #将数据补全后信息返回至原数据
    return tempData

print('done2')

tempData = nbpRandom(tempData, nbp='nisysbp', abp='sysbp')
tempData = nbpRandom(tempData, nbp='nimeanbp', abp='meanbp')
tempData = nbpRandom(tempData, nbp='nidiasbp', abp='diasbp')
print('done3')