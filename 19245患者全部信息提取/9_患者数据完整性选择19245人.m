% Description 
% 程序功能：根据病例数据完整度选择病例
% 程序流程：
%       Step1：计算病例中有效变量的占比
%       Step2：以1%为步长寻找变化率最大的百分比
% 程序运行结果：输出数据完整度性价比最高的新数据集

% DataFile: 病例为1*N的向量
%   new_0709_pat_with_characteristic_notcat.csv   对无创数据进行插补后的数据集

% Output:
%  data 患者数据完整性选择后的新的数据集，每个病例数据完整度均在88%以上
% V1.0 2018/8/28



Data=importdata('new_0709_pat_with_characteristic_notcat.csv');   %读取对无创数据进行插补后的数据集
data=Data.data;

for i=1:size(data,2)
    num(i)=(length(data)-sum(isnan(data(:,i))))/length(data);     %计算各个变量有效数据占病例总数的比例
end

for i=1:size(data,1)
    NotMissing_ratio(i)=(size(data,2)-sum(isnan(data(i,:))))/size(data,2);   %计算各个变量缺失数据占病例总数的比例
end

t=0;
for i=0.01:0.01:1        %以1%的步长计算每个百分比所包含的人数
    t=t+1;
    for_delect_ratio(t)=length(find(    NotMissing_ratio<=1 & NotMissing_ratio>(1-i)   ));
end

t=0;
for i=(length(for_delect_ratio)-1):-1:1
    t=t+1;
    change_ratio(t)=for_delect_ratio(i+1)-for_delect_ratio(i);  %计算其百分比增长人数的变化率，并选择出变化率最高的点对应的百分比（为88%）
end
plot(change_ratio,'LineWidth',2)   %绘制变化率图像
%% 
data=data(find(    NotMissing_ratio<=1 & NotMissing_ratio>(0.88) ),:);   %选择出至少含有88%的有效变量的病例



