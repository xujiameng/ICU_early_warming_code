% Description 
% 程序功能：计算MEWS评分
% 程序流程：
%       Step1：根据MEWS评分标准分别计算各个指标的评分
%       Step2：对各个评分进行整合，得出MEWS评分
% 程序运行结果：输出包括MEWS评分在内的五种传统评分结果

% DataFile: 病例为1*N的向量
%   0731_pingjiazhibiao.csv  其他四种传统评分指标
%   new_0712_pat_with_characteristic_cat_allrunin_complete_in_rf_part_dim_reduction_onehot.csv
%   对所有缺失数据都进行插补后的数据集


% Output:
%  传统评分.xlsx 包括MEWS评分在内的五种传统评分结果
% V1.0 2018/8/28



clear all
clc

Data_pingfen=importdata('0731_pingjiazhibiao.csv');   % 获取其他四种传统评分结果

Data_MEWS=importdata('new_0712_pat_with_characteristic_cat_allrunin_complete_in_rf_part_dim_reduction_onehot.csv');%   获取对所有缺失数据都进行插补后的数据集
data_pingfen=Data_pingfen.data;

data_MEWS=Data_MEWS.data(:,[24,33,109,118,42,51,60,67]);    % 获取计算MEWS评分所需指标

t=0;
for i=1:2:7
    t=t+1;
    for j=1:length(data_MEWS)
        jisuan_MEWS(j,t)=mean(data_MEWS(j,[i,i+1]));
    end
end

%% 以下均是根据MEWAS评分标准编写，可见MEWAS评分标准
pingfen(jisuan_MEWS(:,1)< 40 , 1)=2;
pingfen(       ( jisuan_MEWS(:,1)> 41 & jisuan_MEWS(:,1)< 50 )   , 1)=1;
pingfen(       ( jisuan_MEWS(:,1)> 51 & jisuan_MEWS(:,1)< 100 )   , 1)=0;
pingfen(       ( jisuan_MEWS(:,1)> 101 & jisuan_MEWS(:,1)< 110 )   , 1)=1;
pingfen(       ( jisuan_MEWS(:,1)> 111 & jisuan_MEWS(:,1)< 130 )   , 1)=2;
pingfen(       ( jisuan_MEWS(:,1)> 130 )   , 1)=3;

pingfen(jisuan_MEWS(:,2)< 70 , 2)=3;
pingfen(       ( jisuan_MEWS(:,2)> 71 & jisuan_MEWS(:,2)< 80 )   , 2)=2;
pingfen(       ( jisuan_MEWS(:,2)> 81 & jisuan_MEWS(:,2)< 100 )   , 2)=1;
pingfen(       ( jisuan_MEWS(:,2)> 101 & jisuan_MEWS(:,2)< 199 )   , 2)=0;
pingfen(       ( jisuan_MEWS(:,2)>= 200 )   , 2)=3;

pingfen(jisuan_MEWS(:,3)< 9 , 3)=2;
pingfen(       ( jisuan_MEWS(:,3)> 9 & jisuan_MEWS(:,3)< 14 )   , 3)=0;
pingfen(       ( jisuan_MEWS(:,3)> 15 & jisuan_MEWS(:,3)< 20 )   , 3)=1;
pingfen(       ( jisuan_MEWS(:,3)> 21 & jisuan_MEWS(:,3)< 29 )   , 3)=2;
pingfen(       ( jisuan_MEWS(:,3)>= 30 )   , 3)=3;

pingfen(jisuan_MEWS(:,4)< 35 , 4)=2;
pingfen(       ( jisuan_MEWS(:,4)> 35 & jisuan_MEWS(:,4)< 38.4 )   , 4)=0;
pingfen(       ( jisuan_MEWS(:,4)>= 38.5 )   , 4)=2;

MEWS=sum(pingfen')';   %根据各部分评分计算MEWS评分

pinfen_final=[data_pingfen,MEWS,Data_MEWS.data(:,end)]; %整合包括MEWS评分在内的五种传统评分结果
xlswrite('传统评分.xlsx',pinfen_final);