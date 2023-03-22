clear;clc;close all;
% name='zjl_dy_1sleep_2sit'

fileDir='../slices_20210604/';
fileFolder=fullfile(fileDir);
dirOutput=dir(fullfile(fileFolder,'*.mat'));
dataName={dirOutput.name};
dataName=sort(dataName);

hr1=[];
hr2=[];
br1=[];
br2=[];

absbr1=[];
acchr1=[];
absbr2=[];
acchr2=[];
poshr1=90;
posbr1=16.5;
poshr2=90;
posbr2=16.5;
for i=1:size(dataName,2)
% for i=1:5
    name=char(dataName(i));
    tic;
    load([fileDir,char(dataName(i))]);
    pure_radar1=pca_filter_x4(radar1,2,1,10,40,50);
    pure_radar2=pca_filter_x4(radar2,2,1,10,40,50);
    [~, maxindex1]=max(sum(abs(pure_radar1),1));
    [~, maxindex2]=max(sum(abs(pure_radar2),1));
    [hr_v1,br_v1]=VMDHR202(pure_radar1(:,maxindex1))  %%% VNCMD
    [hr_v2,br_v2]=VMDHR202(pure_radar2(:,maxindex2))  %%% VNCMD
%         [br_v2,hr_v2]=respiration_multi2_vncmd(pure_radar2);   %%% VNCMD
    if hr_v1==-1
        hr_v1=poshr1;
    else
        poshr1=hr_v1;
    end

    if br_v1==-1
        br_v1=posbr1;
    else
        posbr1=br_v1;
    end

    if hr_v2==-1
        hr_v2=poshr2;
    else
        poshr2=hr_v2;
    end

    if br_v2==-1
        br_v2=posbr2;
    else
        posbr2=br_v2;
    end

    hr1=[hr1 hr_v1];
    hr2=[hr2 hr_v2];
    br1=[br1 br_v1];
    br2=[br2 br_v2];
    abserror1=abs(br_v1-breath);
    abserror2=abs(br_v2-breath);
    absbr1=[absbr1 abserror1];
    absbr2=[absbr2 abserror2];

    preacc1=1-abs(hr_v1-oximeter)/oximeter;
    preacc2=1-abs(hr_v2-oximeter)/oximeter;
    acchr1=[acchr1 preacc1];
    acchr2=[acchr2 preacc2];
    disp([...
        'num=',num2str(i),...
        ' abs=',num2str((mean(absbr1)+mean(absbr2))/2),...
        ' per=',num2str((mean(acchr1)+mean(acchr2))/2),...
        ' abs1=',num2str(mean(absbr1)),...
        ' per1=',num2str(mean(acchr1)),...
        ' abs2=',num2str(mean(absbr2)),...
        ' per2=',num2str(mean(acchr2)),...
        ])
    toc;
end
save(['../result/vmd_k7.mat'],'hr1','absbr1','acchr1','hr2','absbr2','acchr2')