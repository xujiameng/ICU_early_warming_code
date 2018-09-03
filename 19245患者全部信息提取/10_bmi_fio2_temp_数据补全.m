% Description 
% �����ܣ���BMI��FIO2�����������������ݲ�����򣬲�ѡ�������ȱʧ�����岹���Ĳ��ֱ���

% �������н������ɶ�BMI,FIO2�������Ĳ岹��ѡ���4������ȱʧ���ݲ岹���ı�����

% DataFile: ����Ϊ1*N������
%    new_0709_pat_with_characteristic_notcat.csv   ���޴����ݽ��в岹������ݼ�

% Output:
%  data��BMI,FIO2�����岹�������
%  Data.textdata(num<0.99)������ȱʧ���ݲ岹���ı���

% V1.0 2018/8/28





clear all
clc
Data=importdata('new_0709_pat_with_characteristic_notcat.xlsx');    %��ȡ�޴����ݲ岹����������ݼ�
data=Data.data;

%% bmi��ǩ���벢ʹ��ƽ��ֵ�岹
bmi_label=ones(length(data),1);        
bmi_label(isnan(data(:,7)))=0;
data(bmi_label==0,7)=mean((data(bmi_label==1,7)));

%% fio2��ǩ�жϲ�ʹ��21%��Ũ�Ȳ岹,�����12Сʱ����ȱʧ��ǰ12Сʱ���ݴ��ڣ�ʹ��ǰ12Сʱ�������
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

data( ( data(:,16)== 0 &   data(:,11)>=0 & isnan(data(:,14))), 14:15) =0;  % ��ԭ����Ϊֻ��һ�������޷����㷽���׼����������䷽���׼����0����
data( ( data(:,25)== 0 &   data(:,20)>=0 & isnan(data(:,23))), 23:24) =0;

qfio2_label=ones(length(data),1);
qfio2_label(isnan(data(:,11)))=0;
hfio2_label=ones(length(data),1);
hfio2_label(isnan(data(:,20)))=0;
sum((isnan(data(:,11)) | isnan(data(:,21))) & data(:,10)==1 )
% find((isnan(data(:,11)) | isnan(data(:,21))) & data(:,10)==1)

%% ��temp�����׼����в岹
data( ( data(:,79)== 0 &   data(:,74)>=0 & isnan(data(:,77))), 77:78) =0;  % ��ԭ����Ϊֻ��һ�������޷����㷽���׼����������䷽���׼����0����
data( ( data(:,70)== 0 &   data(:,65)>=0 & isnan(data(:,68))), 68:69) =0;

%% 
for i=1:size(data,2)
    num(i)=(length(data)-sum(isnan(data(:,i))))/length(data);   %����岹�������������
end

plot(num,'*')
Data.textdata(num<0.99)   %ѡ�������������С��99%�Ĳα���������ģ��Ԥ��Ч���ļ��


%% ����ƴ�� ��154������
% data=[data(:,1:7),bmi_label,data(:,8:end)];
% xlswrite('new_0710_pat_with_characteristic_cat_partrunin.xlsx',data);


