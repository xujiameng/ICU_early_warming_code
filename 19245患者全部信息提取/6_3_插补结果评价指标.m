% Description 
% 程序功能：计算数据补齐结果的部分评价指标
% 程序流程：
%       Step1：获取进行数据补齐的位置
%       Step2：获取数据补齐前后的两组数据，并进行相关性计算及T检验
% 程序运行结果：输出数据补齐结果的评价指标

% DataFile: 病例为1*N的向量
%   临时.xlsx   产生空值的数据集
%   wanzheng.csv   产生空值之前完整的数据集
%   imp_knn.csv    数据补齐后的数据集

% Output:
%  result 包含模型对无创收缩压，舒张压，平均压的部分评价指标
% V1.0 2018/8/28


linshi = importdata('临时.xlsx');  %读取随机产生空值后的数据集
queshi=linshi.data;
nisysbp_loc=isnan(queshi(:,9));    %读取各个变量空值所在位置
nidiasbp_loc=isnan(queshi(:,10));
nimeanbp_loc=isnan(queshi(:,11));
wanzheng=importdata('wanzheng.csv');  %读取生成空值前的完整数据集
e3=wanzheng.data(nisysbp_loc,10);     %读取进行数据补齐位置的原始数据
e4=wanzheng.data(nidiasbp_loc,11);
e5=wanzheng.data(nimeanbp_loc,12);


chabu=importdata('imp_knn.csv');    %读取进行数据补齐后的数据集
e01=chabu.data(nisysbp_loc,10);     %读取进行数据补齐位置的插补数据
e1=chabu.data(nidiasbp_loc,11);
e2=chabu.data(nimeanbp_loc,12);

e01=e01(nisysbp_loc);
e1=e1(nidiasbp_loc);
e2=e2(nimeanbp_loc);



r=corrcoef(e01,e3);     %计算插补前后数据的相关性
[h,p]=ttest(e01,e3);    %对插补前后向量进行T检验，观察是否有显著性的差异
result(1,1)=r(1,2);
result(2,1)=p;

r=corrcoef(e1,e4);  %计算插补前后数据的相关性
[h,p]=ttest(e1,e4);%对插补前后向量进行T检验，观察是否有显著性的差异
result(1,2)=r(1,2);
result(2,2)=p;

r=corrcoef(e2,e5); %计算插补前后数据的相关性
[h,p]=ttest(e2,e5);%对插补前后向量进行T检验，观察是否有显著性的差异
result(1,3)=r(1,2);
result(2,3)=p;

result    %输出数据补齐结果的部分指标
