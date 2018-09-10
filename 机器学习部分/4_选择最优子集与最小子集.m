% Description 
% 程序功能：选择最优特征子集与最小特征子集

% 程序运行结果：输出最优特征子集与最小特征子集位置

% DataFile: 不同模型 N*1
%   'all_ber_mean.csv'   BER平均值
%   'all_ber_std.csv'   BER标准差

% Output:
%  best_num_loc  最优特征子集
%  min_loc  最小特征子集

%Calls：
%	被本函数调用的函数清单
%   xjm.m  获取最小特征子集

% V1.0 2018/9/10



clear all;clc;

MEAN=importdata('all_ber_mean.csv');
FIT=importdata('all_ber_std.csv');
[best_num,best_num_loc]=min(MEAN);    %寻找最小的BER位置及对应的数值

for i=1:size(MEAN,2)
    best_num(i)=best_num(i)+FIT(best_num_loc(i),i);  %将最小BER加上对应的标准差
end
   

for i=1:size(MEAN,2)
    cc=MEAN(1:best_num_loc(i),i) - best_num(i)*ones(size(MEAN(1:best_num_loc(i),i)));   %寻找距离 最小BER加对应标准差 最小的位置
    [min_num(i),min_loc(i)]=min(abs( cc ));  %第一种最小特征子集寻找方法
end

for i=1:size(MEAN,2)
    z(i)=xjm(MEAN(:,i),0.0035);  %第二种最小特征子集计算方法
end
z   %第二种方法得到的最小特征数

for i=1:size(MEAN,2)
% for i=4:4
    figure(i)
    plot(MEAN(:,i),'LineWidth',1.5)

    hold on 
    plot(best_num_loc(i),MEAN(best_num_loc(i),i),'*','LineWidth',2)        %绘制最佳特征子集位置
    plot(min_loc(i),MEAN(min_loc(i),i),'p','LineWidth',2)        %绘制第一种方法得到的最小特征子集位置
    plot(z(i),MEAN(z(i),i),'p','LineWidth',2)       %绘制第二种方法得到的最小特征子集位置
    text(best_num_loc(i),MEAN(best_num_loc(i),i),['(' num2str(best_num_loc(i)) ',' num2str(MEAN(best_num_loc(i),i)) ')'])    
    text(min_loc(i),MEAN(min_loc(i),i),['(' num2str(min_loc(i)) ',' num2str(MEAN(min_loc(i),i)) ')'])
    text(z(i),MEAN(z(i),i),['(' num2str(z(i)) ',' num2str(MEAN(z(i),i)) ')'])  
    
    xlabel('特征量')
    ylabel('十折交叉验证BER平均值')
    legend('十折交叉验证BER平均值','最优特征子集','最小特征子集_1','最小特征子集_2')
end


%% 所使用的函数
% Description 
% 程序功能：选择最小特征子集

% 程序运行结果：输出最小特征子集个数

% DataFile: 
%   data(N*1)：经过特征排序后，机器学习模型十折交叉验证结果的BER平均值
%   yuzhi：BER选择阈值（影响对下降阶段与平缓阶段的选择）

% Output:
%  Fin  最优特征子集的特征量


%Called By：
%	调用本函数的清单
%   4_选择最优子集与最小子集：计算最优特征子集与最小特征子集的特征量

% V1.0 2018/9/10

function [Fin]=xjm(data,yuzhi)  

DA=data;
for i=1:length(data)-1
    if isnan(data(i))==1
        continue
    end
    if length(find( data(i+1:end)>data(i)) )~=0
       z= find( data(i+1:end)>data(i));  %寻找BER一直严格下降的最大子集，其余置为NAN
       z=z+i;
       data(z)=nan;
    end
end
zz=isnan(data);
zzz=find(zz==0);
new_x=zzz;
new_y=data(zzz);

for i=2:length(new_x)
    d_x=new_x(i)-new_x(i-1);
    d_y=new_y(i-1)-new_y(i);
    d(i-1)=d_y/d_x;         %计算滤波后BER平均值的一阶导数，即性价比，某阶段中增加一个特征值能降低多上BER
end

clear t
t=1;
for i=1:length(d)-2
    if sum(find(d(i)<d(i+1:end)))==0  &   sum(   d(i+1:end)  >yuzhi )==0   %限制节点处性价比大于后续位置可能被选择的节点，且导数值小于设定的阈值
       dd = diff(  d(i:(i+ceil( length(d(i:end)))/3  )));

       [~,loc]= min(  abs(  dd  ) );      %寻找节点后数据段前1/3的二阶导数最接近于零的位置  p
       tt=i+loc;
       fin_1(t)=new_x(tt) ;    %BER图像平缓阶段起点
       fin_2(t)=new_x(i)+1 ;   %BER图像下降阶段终点
       t=t+1;
    end
end

if (fin_2(1)-fin_1(1))~=0    
    TTT=abs(fin_2(1)-fin_1(1));  
else if length(fin_2)~=1    %在这里进行一个判断，如果下降阶段终点与平缓阶段起点重合
    TTT=abs(fin_2(2)-fin_1(1));   %则使用二阶导数次接近0的位置代替之前的平缓阶段起点 
    else if length(fin_2)==1   %如果下降阶段后前1/3数据过少
       TTT=ceil(151*0.05);  %使用整个特征量的5%代替原本的长度
        end
    end
end


DATA=DA(fin_1(1)-TTT:fin_1(1));  %提取从关键点1（下降阶段终点）到关键点2（平缓阶段起点）的数据段

[~,loc]= sort(  DA(fin_1(1)-TTT:fin_1(1)),'descend');  %对该数据段数值进行降序排序，并获取其降序前的位置
if  (  abs( DATA(loc(end-1))-DATA(loc(end)) ) /  abs( loc(end-1)-loc(end) )   )<yuzhi/5;      %计算最小BER最小位置与次小位置的一阶导数，如果小于阈值执行以下命令
    F=find(DA==DATA(loc(end-1)));   %如果一接导数小于阈值的1/5，认为增加一小段特征量以减少不足阈值1/5的BER是不值得的，认为次小BER对应的特征子集为最小特征子集
else
    F=find(DA==DATA(loc(end)));%如果一接导数大于阈值的1/5，认为增加一小段特征量以减少大于阈值1/5的BER是值得的，认为最小BER对应的特征子集为最小特征子集
end
Fin=F(1); %输出最小特征子集特征数量

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

