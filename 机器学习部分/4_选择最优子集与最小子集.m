% Description 
% �����ܣ�ѡ�����������Ӽ�����С�����Ӽ�

% �������н����������������Ӽ�����С�����Ӽ�λ��

% DataFile: ��ͬģ�� N*1
%   'all_ber_mean.csv'   BERƽ��ֵ
%   'all_ber_std.csv'   BER��׼��

% Output:
%  best_num_loc  ���������Ӽ�
%  min_loc  ��С�����Ӽ�


% V1.0 2018/8/28



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
    z(i)=xjm(MEAN(:,i));  %�ڶ�����С�����Ӽ����㷽��
end
z   %�ڶ��ַ����õ�����С������

for i=1:size(MEAN,2)
    figure(i)
    plot(MEAN(:,i),'LineWidth',2)

    hold on 
    plot(best_num_loc(i),MEAN(best_num_loc(i),i),'*','LineWidth',2)        %������������Ӽ�λ��
    plot(min_loc(i),MEAN(min_loc(i),i),'p','LineWidth',2)        %���Ƶ�һ�ַ����õ�����С�����Ӽ�λ��
    plot(z(i),MEAN(z(i),i),'p','LineWidth',2)       %���Ƶڶ��ַ����õ�����С�����Ӽ�λ��
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%6
function [Fin]=xjm(data)

DA=data;
for i=1:length(data)-1
    if isnan(data(i))==1
        continue
    end
    if length(find( data(i+1:end)>data(i)) )~=0
       z= find( data(i+1:end)>data(i));  %Ѱ��BERһֱ�½��ĵ㣬������ΪNAN
       z=z+i;
       data(z)=nan;
    end
end
zz=isnan(data);
zzz=find(zz==0);
new_x=zzz;
new_y=data(zzz);

% figure(1)
% plot(new_x,new_y,'-','LineWidth',2)
% hold on 
% plot(new_x,new_y,'*','LineWidth',2)
% plot(DA,'b','LineWidth',1)
% plot([16:28],DA([16:28]),'-','LineWidth',4)
% xlabel('������')
% ylabel('ʮ�۽�����֤BERƽ��ֵ')
% 
% plot([19,27],DA([19,27]),'d','LineWidth',2)
% x=[19,27];y=DA([19,27]);
% for i=1:length(x)
%     text(x(i),y(i),['(' num2str(x(i)) ',' num2str(y(i)) ')'])  
% end



for i=2:length(new_x)
    d_x=new_x(i)-new_x(i-1);
    d_y=new_y(i-1)-new_y(i);
    d(i-1)=d_y/d_x;         %�����˲���BERƽ��ֵ��һ�׵��������Լ۱ȣ�ĳ�׶�������һ������ֵ�ܽ��Ͷ���BER
end

% figure(2)
% plot(d,'p','LineWidth',2)
% hold on 
% plot(d,'-','LineWidth',2)
% plot([8,15,18],d([8,15,18]),'d','LineWidth',5)
% ylabel('һ�׵���')

% zz=diff(d);
% figure(3)
% plot(zz,'p','LineWidth',2)
% hold on 
% plot(zz,'-','LineWidth',2)
% plot([10,16,18],zz([10,16,18]),'d','LineWidth',4)
% ylabel('���׵���')

clear t
t=1;
for i=1:length(d)-1 
    if sum(find(d(i)<d(i+1:end)))==0  &   sum(   d(i+1:end)  >0.005 )==0    %���ƽڵ㴦�Լ۱ȴ��ں���λ�ÿ��ܱ�ѡ��Ľڵ㣬�ҵ���ֵС��0.005
%        [~,loc]= min(  abs(  diff(  d(i:i+ceil( length(d(i:end))/3  )))) );
       dd=  diff(  d(i:i+ceil( length(d(i:end))/4  )));
%        dd(dd<0)=100;
       [~,loc]= min(  abs(  diff(  d(i:i+ceil( length(d(i:end))/4  )))) );      %Ѱ�ҽڵ�����ݶ�ǰ1/4�Ķ��׵�����ӽ������λ��  p
%         [~,loc]= min( dd )
       tt=i+loc;
       fin_1(t)=new_x(tt) ;    %����p��Ӧ����������
       fin_2(t)=new_x(i)+1 ;
       t=t+1;
    end
end

% TTT=ceil(151*0.05);  %�����ڵ�P�������������Ŀռ䳤��
TTT=abs(fin_2(1)-fin_1(1));
DATA=DA(fin_1(1)-TTT:fin_1(1));  %��ȡ�ӹؼ���1���ؼ���2�����ݶ�
[~,loc]= sort(  DA(fin_1(1)-TTT:fin_1(1)),'descend');  %�Ը����ݶ���ֵ���н������򣬲���ȡ�併��ǰ��λ��
if  (  abs( DATA(loc(end-1))-DATA(loc(end)) ) /  abs( loc(end-1)-loc(end) )   )<0.001;      %������СBER��Сλ�����Сλ�õ�һ�׵��������С����ִֵ����������
    F=find(DA==DATA(loc(end-1)));   %���һ�ӵ���С��0.005����Ϊ����һС���������Լ��ٲ���0.005��BER�ǲ�ֵ�õģ���Ϊ��СBER��Ӧ�������Ӽ�Ϊ��С�����Ӽ�
else
    F=find(DA==DATA(loc(end)));%���һ�ӵ�������0.005����Ϊ����һС���������Լ��ٴ���0.005��BER��ֵ�õģ���Ϊ��СBER��Ӧ�������Ӽ�Ϊ��С�����Ӽ�
end

Fin=F(1); %�����С�����Ӽ���������
end