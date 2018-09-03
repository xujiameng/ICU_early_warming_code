% Description 
% 程序功能：选择最优特征子集与最小特征子集

% 程序运行结果：输出最优特征子集与最小特征子集位置

% DataFile: 不同模型 N*1
%   'all_ber_mean.csv'   BER平均值
%   'all_ber_std.csv'   BER标准差

% Output:
%  best_num_loc  最优特征子集
%  min_loc  最小特征子集


% V1.0 2018/8/28



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
    z(i)=xjm(MEAN(:,i));  %第二种最小特征子集计算方法
end
z   %第二种方法得到的最小特征数

for i=1:size(MEAN,2)
    figure(i)
    plot(MEAN(:,i),'LineWidth',2)

    hold on 
    plot(best_num_loc(i),MEAN(best_num_loc(i),i),'*','LineWidth',2)        %绘制最佳特征子集位置
    plot(min_loc(i),MEAN(min_loc(i),i),'p','LineWidth',2)        %绘制第一种方法得到的最小特征子集位置
    plot(z(i),MEAN(z(i),i),'p','LineWidth',2)       %绘制第二种方法得到的最小特征子集位置
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%6
function [Fin]=xjm(data)

DA=data;
for i=1:length(data)-1
    if isnan(data(i))==1
        continue
    end
    if length(find( data(i+1:end)>data(i)) )~=0
       z= find( data(i+1:end)>data(i));  %寻找BER一直下降的点，其余置为NAN
       z=z+i;
       data(z)=nan;
    end
end
zz=isnan(data);
zzz=find(zz==0);
new_x=zzz;
new_y=data(zzz);

% figure(1)
% plot(new_x,new_y,'-')
% hold on 
% plot(new_x,new_y,'*')

for i=2:length(new_x)
    d_x=new_x(i)-new_x(i-1);
    d_y=new_y(i-1)-new_y(i);
    d(i-1)=d_y/d_x;         %计算滤波后BER平均值的一阶导数，即性价比，某阶段中增加一个特征值能降低多上BER
end
% figure(2)
% plot(d,'p')
% hold on 
% plot(d,'-')
% 
% figure(3)
% plot(diff(d),'p')
% hold on 
% plot(diff(d),'-')

clear t
t=1;
for i=1:length(d)-1 
    if sum(find(d(i)<d(i+1:end)))==0  &   sum(   d(i+1:end)  >0.005 )==0      %限制节点处性价比大于后续位置可能被选择的节点，且导数值小于0.005
%        [~,loc]= min(  abs(  diff(  d(i:i+ceil( length(d(i:end))/3  )))) );
       [~,loc]= min(  abs(  diff(  d(i:i+ceil( length(d(i:end))/4  )))) );      %寻找节点后数据段前1/4的二阶导数最接近于零的位置  p
       tt=i+loc;
       fin_1(t)=new_x(tt)+1 ;    %计算p对应的特征数量
%        fin_2(t)=new_x(i)+1 ;
       t=t+1;
    end
end

TTT=ceil(151*0.05);  %创建在点P附近继续搜索的空间长度
% TTT=10;
DATA=DA(fin_1(1)-TTT:fin_1(1));  %提取以点P为中心的二次检索的数据段
[~,loc]= sort(  DA(fin_1(1)-TTT:fin_1(1)),'descend');  %对该数据段数值进行降序排序，并获取其降序前的位置
if  (  abs( DATA(loc(end-1))-DATA(loc(end)) ) /  abs( loc(end-1)-loc(end) )   )<0.005;      %计算最小BER最小位置与次小位置的一阶导数，如果小于阈值执行以下命令
    F=find(DA==DATA(loc(end-1)));   %如果一接导数小于0.005，认为增加一小段特征量以减少不足0.005的BER是不值得的，认为次小BER对应的特征子集为最小特征子集
else
    F=find(DA==DATA(loc(end)));%如果一接导数大于0.005，认为增加一小段特征量以减少大于0.005的BER是值得的，认为最小BER对应的特征子集为最小特征子集
end

Fin=F(1); %输出最小特征子集特征数量
end




