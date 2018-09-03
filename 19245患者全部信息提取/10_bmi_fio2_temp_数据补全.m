% Description 
% 程序功能：对BMI，FIO2两个特殊变量设计数据补齐规则，并选择出进行缺失变量插补检测的部分变量

% 程序运行结果：完成对BMI,FIO2两变量的插补。选择出4个进行缺失数据插补检测的变量。

% DataFile: 病例为1*N的向量
%    new_0709_pat_with_characteristic_notcat.csv   对无创数据进行插补后的数据集

% Output:
%  data：BMI,FIO2变量插补后的数据
%  Data.textdata(num<0.99)：进行缺失数据插补检测的变量

% V1.0 2018/8/28





clear all
clc
Data=importdata('new_0709_pat_with_characteristic_notcat.xlsx');    %读取无创数据插补完整后的数据集
data=Data.data;

%% bmi标签加入并使用平均值插补
bmi_label=ones(length(data),1);        
bmi_label(isnan(data(:,7)))=0;
data(bmi_label==0,7)=mean((data(bmi_label==1,7)));

%% fio2标签判断并使用21%氧浓度插补,如果后12小时数据缺失，前12小时数据存在，使用前12小时数据填补。
qfio2_label=ones(length(data),1);
qfio2_label(isnan(data(:,11)))=0;

hfio2_label=ones(length(data),1);
hfio2_label(isnan(data(:,20)))=0;
data((hfio2_label==0&qfio2_label==1),20:28)=data((hfio2_label==0&qfio2_label==1),11:19);
hfio2_label=ones(length(data),1);
hfio2_label(isnan(data(:,20)))=0;

data(data(:,10)==0,11:28)=21;
data(data(:,10)==0,14:16)=0;
data(data(:,10)==0,23:25)=0;

data( (data(:,10)==1 & qfio2_label==0 & hfio2_label==1)    ,11:28)=21;
data( (data(:,10)==1 & qfio2_label==0 & hfio2_label==1)    ,14:16)=0;
data( (data(:,10)==1 & qfio2_label==0 & hfio2_label==1)    ,23:25)=0;

data( ( data(:,16)== 0 &   data(:,11)>=0 & isnan(data(:,14))), 14:15) =0;  % 对原本因为只有一个数据无法计算方差标准差的特征，其方差，标准差用0代替
data( ( data(:,25)== 0 &   data(:,20)>=0 & isnan(data(:,23))), 23:24) =0;

qfio2_label=ones(length(data),1);
qfio2_label(isnan(data(:,11)))=0;
hfio2_label=ones(length(data),1);
hfio2_label(isnan(data(:,20)))=0;
sum((isnan(data(:,11)) | isnan(data(:,21))) & data(:,10)==1 )
% find((isnan(data(:,11)) | isnan(data(:,21))) & data(:,10)==1)

%% 对temp方差标准差进行插补
data( ( data(:,79)== 0 &   data(:,74)>=0 & isnan(data(:,77))), 77:78) =0;  % 对原本因为只有一个数据无法计算方差标准差的特征，其方差，标准差用0代替
data( ( data(:,70)== 0 &   data(:,65)>=0 & isnan(data(:,68))), 68:69) =0;

%% 
for i=1:size(data,2)
    num(i)=(length(data)-sum(isnan(data(:,i))))/length(data);   %计算插补后的数据完整度
end

plot(num,'*')
Data.textdata(num<0.99)   %选择出数据完整度小于99%的参变量，用于模型预测效果的检测


%% 数据拼接 共154个特征
% data=[data(:,1:7),bmi_label,data(:,8:end)];
% xlswrite('new_0710_pat_with_characteristic_cat_partrunin.xlsx',data);


