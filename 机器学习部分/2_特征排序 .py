# Description 
# 程序功能：根据数据补全且进行显著性检验后的数据集进行特征权重计算与特征排序
# 程序流程：
#       Step1：计算三种算法下的特征权重
#       Step2：根据不同的特征权重计算最终的特征权重
# 程序运行结果：输出各个变量最终的特征权重
#
# DataFile: 病例为1*N的向量
#   new_0712_pat_with_characteristic_cat_allrunin_complete_in_rf_part_dim_reduction_onehot.csv  对数据进行数据补齐后的完整数据
#
# Output:
#    result_w  各个变量最终的特征权重
# V1.0 2018/8/28



import pandas as pd
from sklearn import cross_validation
import numpy as np
from sklearn import preprocessing
from skfeature.function.similarity_based import fisher_score
from skfeature.function.similarity_based import reliefF
from skfeature.function.statistical_based import gini_index


def preprocess(dataset):#将特征值规范化到[0,1]之间
    min_max_scaler=preprocessing.MinMaxScaler()
    X_train01=min_max_scaler.fit_transform(dataset)  
    return X_train01


comtest= pd.read_csv("new_0718_for_order.csv")

def weight():
#    x_train, datamat, y_train,labelmat = cross_validation.train_test_split(comtest.iloc[0:len(comtest),1:comtest.shape[1]-1],comtest.iloc[0:len(comtest),-1], test_size = 0.2,random_state = j) 
#    datamat=np.array(datamat,dtype=np.float)
#    labelmat=np.array(labelmat,dtype=np.int)
    datamat=np.array(comtest.iloc[0:len(comtest),1:comtest.shape[1]-1],dtype=np.float)  #提取病例数据及其标签
    labelmat=np.array(comtest.iloc[0:len(comtest),-1],dtype=np.int)
    datamat=preprocess(datamat)
    for i in range(len(labelmat)):
        if labelmat[i]==0:
            labelmat[i]=-1;#adaboost只能区分-1和1的标签
            
    Relief = reliefF.reliefF(datamat, labelmat)   #计算Relieff下的特征权重
    print('Relief, 第%s次验证 '%(1))
    Fisher= fisher_score.fisher_score(datamat, labelmat)  #计算fisher下的特征权重
    print('Fisher, 第%s次验证 '%(1))
    gini= gini_index.gini_index(datamat,labelmat)  #计算gini下的特征权重
    gini=-gini
    print('gini, 第%s次验证 '%(1))
    print("done_ %s" )
    return Relief, Fisher, gini

R=[]
F=[]
g=[]

Relief, Fisher, gini = weight()
    

FSscore=np.column_stack((Relief,Fisher,gini))#合并三个分数
FSscore = preprocess(FSscore)  # 对三个得分进行0-1标准化
result_w=np.sum(FSscore,axis=1)   # 获取最终得分