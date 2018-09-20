# ICU_early_warming_code
<br>
<br>
　该文件包含代码与数据，共分为三个部分，主要使用SQL与python进行编程，考虑到matlab对矩阵处理的灵活性，小部分功能使用matlab实现<br> <br>

## 第一部分，患者信息提取<br>
　　（１）　从ＭＩＭＩＣ　ＩＩＩ数据库中选择满足条件的病例，并对部分完整程度较低的病例数据进行删除，剩余患者１９２４５人，其中１６５５２人存活，２６９３人死亡，随后进行数据补全并计算相应的衍生变量。<br><br><br>

## 第二部分，机器学习模型构建与预测<br>
　　（１）　首先对病例特征进行显著性检验，删除无显著性的９个特征，剩余１５１个特征；<br>
　　（２）　随后使用结合ＲｅｌｉｅｆＦ，Ｆｉｓｈｅｒ，Ｇｉｎｉ三种方法的综合排序方法对病例特征进行排序；　<br>
　　（３）　根据特征排序结果使用８种机器学习模型按特征量逐次递增的方式，通过十折交叉验证计算不同特征子集的预测结果的平均值与标准差；<br>
　　（４）　不同子集预测结果的ＢＥＲ值选择出最优特征子集，最小特征子集的特征量；<br>
　　（５）　通过训练集与验证集得出最优分类阈值，并将其视为参数使用到测试集的预测中，在此最优特征选择标准为使得敏感性与特异性最为相近。同时计算默认阈值下的预测结果<br><br>
  
　
## 第三部分，使用传统评分系统进行患者生死预测<br>
 　　（１）　除去ＭＥＷＳ评分外，ＳＡＰＳＩＩＩ，ＳＯＦＡ，ＡＰＳＩＩＩ，ＯＡＳＩＳ，均能从ＭＩＭＩＣ　ＩＩＩ数据库中直接提取。<br>
 　　（２）　目前选择传统评分系统的最优分类阈值的方法有两个：<br>
 　　 　　 　　ｉ）　(舍弃)通过ＲＯＣ曲线的横纵坐标计算集合　ＴＰＲ＋（１－ＦＰＲ）－１，该集合分布与ＭＣＣ相同，只需找到使其相应原始达到最大的元素即可；<br>
 　　 　　 　ｉｉ）　计算不同阈值下的敏感性与特异性，选择其中使得敏感性最接近特异性的阈值作为最优分类阈值。
 
<br><br><br>


# 函数功能描述<br>
## 19245患者全部信息提取　　文件夹<br>
 　　（１）　1_28612患者提取：　提取满足以下条件的患者，即以下患者满足模型要求，但尚未进行数据预处理<br>
 　　（２）　2_提取血压测量值，为数据补齐做准备q12：　提取患者前12小时无创血压数据及有创血压数据<br>
 　　（３）　3_提取血压测量值，为数据补齐做准备h12：　提取患者进入ICU后12-24小时无创血压数据及有创血压数据<br>
 　　（４）　4_提取血压测量值，为数据补齐做准备all：　提取患者进入ICU第一天无创血压数据及有创血压数据<br>  
 　　（５）　5_提取血压测量值，为数据补齐做准备_插补用信息整合：　整合无创血压、有创血压与其它变量，为接下来的无创血压数据插补做数据准备<br>
 　　（６）　6_1_随机森林对无创数据的插补：　使用随机森林对缺失的无创血压数据进行插补，使用参数有gender','icutype','age','vent'<br>  
 　　　　　　6_2_其它对无创数据的插补与部分评价指标：　使用平均值，中位数，众数，KNN对缺失数据进行插补，并对插补结果做出相应评价<br> 
 　　　　　　6_3_插补结果评价指标：　计算数据补齐结果的部分评价指标：显著性系数，相关性系数<br> 
 　　（７）　7_1_28612患者前12h血压信息准备：　计算患者进入ICU前12小时无创血压时序数据的衍生变量<br> 
 　　　　　　7_2_28612患者后12h血压信息准备：　计算患者进入ICU 12-24小时无创血压时序数据的衍生变量<br>
 　　　　　　7_3_28612患者前12h其它时序数据信息准备：　计算患者进入ICU前12小时除无创血压外其它时序数据的衍生变量<br>
 　　　　　　7_4_28612患者后12h其它时序数据信息准备：　计算患者进入ICU 12~24 小时除无创血压外其它时序数据的衍生变量<br>
 　　　　　　7_5_28612患者前12h生理变量准备：　汇总患者进入ICU前12小时生理数据的衍生变量<br>
 　　　　　　7_6_28612患者后12h生理变量准备：　计算患者进入ICU 12-24 小时生理数据的衍生变量<br>
 　　　　　　7_7_28612患者生理变量整合：　汇总患者进入ICU第一天生理数据的衍生变量<br>
 　　　　　　7_8_28612患者人口统计变量等准备：　计算患者进入ICU第一天人口统计变量等数据的衍生变量<br>
 　　（８）　8_28612患者信息提取：　整合所有生理变量，人口统计变量等<br>
 　　（９）　9_患者数据完整性选择19245人：　根据病例数据完整度选择信息较为有效的病例<br>
 　　（１０）10_bmi_fio2_temp_数据补全：　对BMI，FIO2两个特殊变量设计数据补齐规则，并选择出进行缺失变量插补检测的部分变量<br>
 　　（１１）11_对缺失的特征值进行插补：　使用随机森林对缺失数据进行插补（最后的数据补全）<br><br>

## 机器学习部分　　文件夹<br>
 　　（１）　1_显著性检验使用SPSS完成<br>
 　　（２）　2_特征排序：　根据数据补全且进行显著性检验后的数据集进行特征权重计算与特征排序<br>
 　　（３）　3_计算不同子集于验证集上的预测结果（默认分类阈值为0.5）：　计算不同机器学习模型不同特征子集下，验证集上的预测结果评价指标<br>
 　　（４）　4_选择最优子集与最小子集：　选择最优特征子集与最小特征子集<br>
 　　（５）　5_计算三种子集下的最优分类阈值与默认阈值的预测结果：　通过训练集与验证集获取最优分类阈值，并应用到不同子集的测试集上进行结果预测<br><br>

## 传统评分部分　　文件夹<br>
 　　（１）　1_MEWS评分计算：　根据患者心率，收缩压，呼吸，体温等计算MEWS评分<br>
 　　（２）　2_传统评分最优分类阈值的获取：　根据训练集与验证集获取各个传统评分系统对应的最佳分类阈值<br>
 　　（３）　3_传统评分在最优分类阈值下测试集预测结果计算：　在获取最佳分类阈值的基础上在测试集上进行结果预测





