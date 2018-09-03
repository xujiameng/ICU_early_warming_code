

#* Description 
#* 程序功能：使用平均值，中位数，众数，KNN对缺失数据进行插补
#*程序流程：
#*       Step1：在数据集中随机产生缺失数据，对缺失位置数据进行数据补全
#*      Step2：对补全后的数据情况进行评价
#* 程序运行结果：输出数据补全后的数据集及对应的评价结果

#* DataFile: 病例为1*N的向量
#*  wanzheng.csv   含有完整数据的数据集

#* Output:
#*  imp_knn       KNN 的插补结果
#*  imp_median  中位数的插补结果
#*  imp_mean   平均值 的插补结果
#*  zhongshu  众数 的插补结果 
#* error     数据补齐结果的评价结果

#* V1.0 2018/8/28





#*********导入所需包************#
library(lattice)
library(survival)
library(Formula)
library(ggplot2)
library(DMwR)
library(Hmisc)
library(magrittr)
library(dplyr)



data_complete <- read.csv("C:/Users/佳盟/Desktop/死亡率提纲/wanzheng.csv")
actual <- data_complete

data_complete$nisysbp[sample(nrow(data_complete), 8058)] <- NA    #  在完整的数据中无创变量部分随机产生1/3的缺失数据
data_complete$nidiasbp[sample(nrow(data_complete), 8058)] <- NA 
data_complete$nimeanbp[sample(nrow(data_complete), 8058)] <- NA 



#**********依次用knn、中位数、平均值、众数对缺失数据进行插补************#
imp_knn <- knnImputation(data_complete, k = 10, scale = T, meth = "weighAvg", distData = NULL)#knn
imp_median <- centralImputation(data_complete)#非缺失样本的中位数插值
imp_mean<- (data_complete %>% mutate_all(impute,mean))#非缺失样本的mena插值
zhongshu <- function(x)
{
  tmp<-(as.numeric(names(table(x))[table(x)==max(table(x))]))
  return(mean(tmp))#多个数的频率是一样的
}
imp_zhongshu <- (data_complete %>% mutate_all(impute,zhongshu))#非缺失样本的众数插值

#**********计算插补后的误差************#
imp <- imp_zhongshu#这里不太智能，需要手动将imp_knn等这些插补结果赋值给imp再进行计算，手动记录误差结果
nisysbp<-regr.eval(actual$nisysbp, imp$nisysbp)
nidiasbp<-regr.eval(actual$nidiasbp, imp$nidiasbp)
nimeanbp<-regr.eval(actual$nimeanbp, imp$nimeanbp)

error<- cbind(nisysbp,nidiasbp,nimeanbp)









imp <- read.csv("C:/Users/佳盟/Desktop/死亡率提纲/xgb0705.csv")
nisysbp<-regr.eval(actual$nisysbp, imp$nisysbp)
nidiasbp<-regr.eval(actual$nidiasbp, imp$nidiasbp)
nimeanbp<-regr.eval(actual$nimeanbp, imp$nimeanbp)
error<- cbind(nisysbp,nidiasbp,nimeanbp)