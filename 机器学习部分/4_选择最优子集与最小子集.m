% Description 
% �����ܣ�ѡ�����������Ӽ�����С�����Ӽ�

% �������н����������������Ӽ�����С�����Ӽ�λ��

% DataFile: ��ͬģ�� N*1
%   'all_ber_mean.csv'   BERƽ��ֵ
%   'all_ber_std.csv'   BER��׼��

% Output:
%  best_num_loc  ���������Ӽ�
%  min_loc  ��С�����Ӽ�

%Calls��
%	�����������õĺ����嵥
%   xjm.m  ��ȡ��С�����Ӽ�

% V1.0 2018/9/10



clear all;clc;

MEAN=importdata('all_ber_mean.csv');
FIT=importdata('all_ber_std.csv');
[best_num,best_num_loc]=min(MEAN);    %Ѱ����С��BERλ�ü���Ӧ����ֵ

for i=1:size(MEAN,2)
    best_num(i)=best_num(i)+FIT(best_num_loc(i),i);  %����СBER���϶�Ӧ�ı�׼��
end
   

for i=1:size(MEAN,2)
    cc=MEAN(1:best_num_loc(i),i) - best_num(i)*ones(size(MEAN(1:best_num_loc(i),i)));   %Ѱ�Ҿ��� ��СBER�Ӷ�Ӧ��׼�� ��С��λ��
    [min_num(i),min_loc(i)]=min(abs( cc ));  %��һ����С�����Ӽ�Ѱ�ҷ���
end

for i=1:size(MEAN,2)
    z(i)=xjm(MEAN(:,i),0.0035);  %�ڶ�����С�����Ӽ����㷽��
end
z   %�ڶ��ַ����õ�����С������

for i=1:size(MEAN,2)
% for i=4:4
    figure(i)
    plot(MEAN(:,i),'LineWidth',1.5)

    hold on 
    plot(best_num_loc(i),MEAN(best_num_loc(i),i),'*','LineWidth',2)        %������������Ӽ�λ��
    plot(min_loc(i),MEAN(min_loc(i),i),'p','LineWidth',2)        %���Ƶ�һ�ַ����õ�����С�����Ӽ�λ��
    plot(z(i),MEAN(z(i),i),'p','LineWidth',2)       %���Ƶڶ��ַ����õ�����С�����Ӽ�λ��
    text(best_num_loc(i),MEAN(best_num_loc(i),i),['(' num2str(best_num_loc(i)) ',' num2str(MEAN(best_num_loc(i),i)) ')'])    
    text(min_loc(i),MEAN(min_loc(i),i),['(' num2str(min_loc(i)) ',' num2str(MEAN(min_loc(i),i)) ')'])
    text(z(i),MEAN(z(i),i),['(' num2str(z(i)) ',' num2str(MEAN(z(i),i)) ')'])  
    
    xlabel('������')
    ylabel('ʮ�۽�����֤BERƽ��ֵ')
    legend('ʮ�۽�����֤BERƽ��ֵ','���������Ӽ�','��С�����Ӽ�_1','��С�����Ӽ�_2')
end


%% ��ʹ�õĺ���
% Description 
% �����ܣ�ѡ����С�����Ӽ�

% �������н���������С�����Ӽ�����

% DataFile: 
%   data(N*1)��������������󣬻���ѧϰģ��ʮ�۽�����֤�����BERƽ��ֵ
%   yuzhi��BERѡ����ֵ��Ӱ����½��׶���ƽ���׶ε�ѡ��

% Output:
%  Fin  ���������Ӽ���������


%Called By��
%	���ñ��������嵥
%   4_ѡ�������Ӽ�����С�Ӽ����������������Ӽ�����С�����Ӽ���������

% V1.0 2018/9/10

function [Fin]=xjm(data,yuzhi)  

DA=data;
for i=1:length(data)-1
    if isnan(data(i))==1
        continue
    end
    if length(find( data(i+1:end)>data(i)) )~=0
       z= find( data(i+1:end)>data(i));  %Ѱ��BERһֱ�ϸ��½�������Ӽ���������ΪNAN
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
    d(i-1)=d_y/d_x;         %�����˲���BERƽ��ֵ��һ�׵��������Լ۱ȣ�ĳ�׶�������һ������ֵ�ܽ��Ͷ���BER
end

clear t
t=1;
for i=1:length(d)-2
    if sum(find(d(i)<d(i+1:end)))==0  &   sum(   d(i+1:end)  >yuzhi )==0   %���ƽڵ㴦�Լ۱ȴ��ں���λ�ÿ��ܱ�ѡ��Ľڵ㣬�ҵ���ֵС���趨����ֵ
       dd = diff(  d(i:(i+ceil( length(d(i:end)))/3  )));

       [~,loc]= min(  abs(  dd  ) );      %Ѱ�ҽڵ�����ݶ�ǰ1/3�Ķ��׵�����ӽ������λ��  p
       tt=i+loc;
       fin_1(t)=new_x(tt) ;    %BERͼ��ƽ���׶����
       fin_2(t)=new_x(i)+1 ;   %BERͼ���½��׶��յ�
       t=t+1;
    end
end

if (fin_2(1)-fin_1(1))~=0    
    TTT=abs(fin_2(1)-fin_1(1));  
else if length(fin_2)~=1    %���������һ���жϣ�����½��׶��յ���ƽ���׶�����غ�
    TTT=abs(fin_2(2)-fin_1(1));   %��ʹ�ö��׵����νӽ�0��λ�ô���֮ǰ��ƽ���׶���� 
    else if length(fin_2)==1   %����½��׶κ�ǰ1/3���ݹ���
       TTT=ceil(151*0.05);  %ʹ��������������5%����ԭ���ĳ���
        end
    end
end


DATA=DA(fin_1(1)-TTT:fin_1(1));  %��ȡ�ӹؼ���1���½��׶��յ㣩���ؼ���2��ƽ���׶���㣩�����ݶ�

[~,loc]= sort(  DA(fin_1(1)-TTT:fin_1(1)),'descend');  %�Ը����ݶ���ֵ���н������򣬲���ȡ�併��ǰ��λ��
if  (  abs( DATA(loc(end-1))-DATA(loc(end)) ) /  abs( loc(end-1)-loc(end) )   )<yuzhi/5;      %������СBER��Сλ�����Сλ�õ�һ�׵��������С����ִֵ����������
    F=find(DA==DATA(loc(end-1)));   %���һ�ӵ���С����ֵ��1/5����Ϊ����һС���������Լ��ٲ�����ֵ1/5��BER�ǲ�ֵ�õģ���Ϊ��СBER��Ӧ�������Ӽ�Ϊ��С�����Ӽ�
else
    F=find(DA==DATA(loc(end)));%���һ�ӵ���������ֵ��1/5����Ϊ����һС���������Լ��ٴ�����ֵ1/5��BER��ֵ�õģ���Ϊ��СBER��Ӧ�������Ӽ�Ϊ��С�����Ӽ�
end
Fin=F(1); %�����С�����Ӽ���������

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

