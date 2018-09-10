% Description 
% �����ܣ�����MEWS����
% �������̣�
%       Step1������MEWS���ֱ�׼�ֱ�������ָ�������
%       Step2���Ը������ֽ������ϣ��ó�MEWS����
% �������н�����������MEWS�������ڵ����ִ�ͳ���ֽ��

% DataFile: ����Ϊ1*N������
%   0731_pingjiazhibiao.csv  �������ִ�ͳ����ָ��
%   new_0712_pat_with_characteristic_cat_allrunin_complete_in_rf_part_dim_reduction_onehot.csv
%   ������ȱʧ���ݶ����в岹������ݼ�


% Output:
%  ��ͳ����.xlsx ����MEWS�������ڵ����ִ�ͳ���ֽ��
% V1.0 2018/8/28



clear all
clc

Data_pingfen=importdata('0731_pingjiazhibiao.csv');   % ��ȡ�������ִ�ͳ���ֽ��

Data_MEWS=importdata('new_0712_pat_with_characteristic_cat_allrunin_complete_in_rf_part_dim_reduction_onehot.csv');%   ��ȡ������ȱʧ���ݶ����в岹������ݼ�
data_pingfen=Data_pingfen.data;

data_MEWS=Data_MEWS.data(:,[24,33,109,118,42,51,60,67]);    % ��ȡ����MEWS��������ָ��

t=0;
for i=1:2:7
    t=t+1;
    for j=1:length(data_MEWS)
        jisuan_MEWS(j,t)=mean(data_MEWS(j,[i,i+1]));
    end
end

%% ���¾��Ǹ���MEWAS���ֱ�׼��д���ɼ�MEWAS���ֱ�׼
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

MEWS=sum(pingfen')';   %���ݸ��������ּ���MEWS����

pinfen_final=[data_pingfen,MEWS,Data_MEWS.data(:,end)]; %���ϰ���MEWS�������ڵ����ִ�ͳ���ֽ��
xlswrite('��ͳ����.xlsx',pinfen_final);