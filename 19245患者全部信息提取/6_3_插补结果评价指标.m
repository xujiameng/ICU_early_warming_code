% Description 
% �����ܣ��������ݲ������Ĳ�������ָ��
% �������̣�
%       Step1����ȡ�������ݲ����λ��
%       Step2����ȡ���ݲ���ǰ����������ݣ�����������Լ��㼰T����
% �������н����������ݲ�����������ָ��

% DataFile: ����Ϊ1*N������
%   ��ʱ.xlsx   ������ֵ�����ݼ�
%   wanzheng.csv   ������ֵ֮ǰ���������ݼ�
%   imp_knn.csv    ���ݲ��������ݼ�

% Output:
%  result ����ģ�Ͷ��޴�����ѹ������ѹ��ƽ��ѹ�Ĳ�������ָ��
% V1.0 2018/8/28


linshi = importdata('��ʱ.xlsx');  %��ȡ���������ֵ������ݼ�
queshi=linshi.data;
nisysbp_loc=isnan(queshi(:,9));    %��ȡ����������ֵ����λ��
nidiasbp_loc=isnan(queshi(:,10));
nimeanbp_loc=isnan(queshi(:,11));
wanzheng=importdata('wanzheng.csv');  %��ȡ���ɿ�ֵǰ���������ݼ�
e3=wanzheng.data(nisysbp_loc,10);     %��ȡ�������ݲ���λ�õ�ԭʼ����
e4=wanzheng.data(nidiasbp_loc,11);
e5=wanzheng.data(nimeanbp_loc,12);


chabu=importdata('imp_knn.csv');    %��ȡ�������ݲ��������ݼ�
e01=chabu.data(nisysbp_loc,10);     %��ȡ�������ݲ���λ�õĲ岹����
e1=chabu.data(nidiasbp_loc,11);
e2=chabu.data(nimeanbp_loc,12);

e01=e01(nisysbp_loc);
e1=e1(nidiasbp_loc);
e2=e2(nimeanbp_loc);



r=corrcoef(e01,e3);     %����岹ǰ�����ݵ������
[h,p]=ttest(e01,e3);    %�Բ岹ǰ����������T���飬�۲��Ƿ��������ԵĲ���
result(1,1)=r(1,2);
result(2,1)=p;

r=corrcoef(e1,e4);  %����岹ǰ�����ݵ������
[h,p]=ttest(e1,e4);%�Բ岹ǰ����������T���飬�۲��Ƿ��������ԵĲ���
result(1,2)=r(1,2);
result(2,2)=p;

r=corrcoef(e2,e5); %����岹ǰ�����ݵ������
[h,p]=ttest(e2,e5);%�Բ岹ǰ����������T���飬�۲��Ƿ��������ԵĲ���
result(1,3)=r(1,2);
result(2,3)=p;

result    %������ݲ������Ĳ���ָ��
