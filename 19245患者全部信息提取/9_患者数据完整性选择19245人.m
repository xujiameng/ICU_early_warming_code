% Description 
% �����ܣ����ݲ�������������ѡ����
% �������̣�
%       Step1�����㲡������Ч������ռ��
%       Step2����1%Ϊ����Ѱ�ұ仯�����İٷֱ�
% �������н������������������Լ۱���ߵ������ݼ�

% DataFile: ����Ϊ1*N������
%   new_0709_pat_with_characteristic_notcat.csv   ���޴����ݽ��в岹������ݼ�

% Output:
%  data ��������������ѡ�����µ����ݼ���ÿ���������������Ⱦ���88%����
% V1.0 2018/8/28



Data=importdata('new_0709_pat_with_characteristic_notcat.csv');   %��ȡ���޴����ݽ��в岹������ݼ�
data=Data.data;

for i=1:size(data,2)
    num(i)=(length(data)-sum(isnan(data(:,i))))/length(data);     %�������������Ч����ռ���������ı���
end

for i=1:size(data,1)
    NotMissing_ratio(i)=(size(data,2)-sum(isnan(data(i,:))))/size(data,2);   %�����������ȱʧ����ռ���������ı���
end

t=0;
for i=0.01:0.01:1        %��1%�Ĳ�������ÿ���ٷֱ�������������
    t=t+1;
    for_delect_ratio(t)=length(find(    NotMissing_ratio<=1 & NotMissing_ratio>(1-i)   ));
end

t=0;
for i=(length(for_delect_ratio)-1):-1:1
    t=t+1;
    change_ratio(t)=for_delect_ratio(i+1)-for_delect_ratio(i);  %������ٷֱ����������ı仯�ʣ���ѡ����仯����ߵĵ��Ӧ�İٷֱȣ�Ϊ88%��
end
plot(change_ratio,'LineWidth',2)   %���Ʊ仯��ͼ��
%% 
data=data(find(    NotMissing_ratio<=1 & NotMissing_ratio>(0.88) ),:);   %ѡ������ٺ���88%����Ч�����Ĳ���



