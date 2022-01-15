%% Cultural Algorithm Image Quantization - Created in 16 Jan 2022 by Seyed Muhammad Hossein Mousavi
% This code, takes a input image and applies Otsu’s thresholding method 
% based on user's threshold levels on
% image for quantization. Then the achieved vector from preview step sends
% to Cultural Algorithm cycle to be fitted even better. You can change
% parameters and input image as you desired. 
% ------------------------------------------------ 
% Feel free to contact me if you find any problem using the code: 
% Author: SeyedMuhammadHosseinMousavi
% My Email: mosavi.a.i.buali@gmail.com 
% My Google Scholar: https://scholar.google.com/citations?user=PtvQvAQAAAAJ&hl=en 
% My GitHub: https://github.com/SeyedMuhammadHosseinMousavi?tab=repositories 
% My ORCID: https://orcid.org/0000-0001-6906-2152 
% My Scopus: https://www.scopus.com/authid/detail.uri?authorId=57193122985 
% My MathWorks: https://www.mathworks.com/matlabcentral/profile/authors/9763916#
% my RG: https://www.researchgate.net/profile/Seyed-Mousavi-17
% ------------------------------------------------ 
% Hope it help you, enjoy the code and wish me luck :)

%% Clearing Things
clc
clear
close all
warning ('off');

%% Data Load and Preparation
% Number of Threshold Levels
thresholdlvl=13;
%
%Loading Input Image
I = imread('eva.jpg');
% Convert To Gray
I=rgb2gray(I);
% Basic Multilevel Image Thresholds Using Otsu’s Method
Data = multithresh(I,thresholdlvl);
Data=Data';
Data=double(Data);
% Creating Inputs and Targets
Delays = [1];
[Inputs, Targets] = MakeTheTimeSeries(Data',Delays);
data.Inputs=Inputs;
data.Targets=Targets;
% Making Data
Inputs=data.Inputs';
Targets=data.Targets';
Targets=Targets(:,1);
nSample=size(Inputs,1);
% Creating Train Vector
pTrain=1.0;
nTrain=round(pTrain*nSample);
TrainInputs=Inputs(1:nTrain,:);
TrainTargets=Targets(1:nTrain,:);
TestInputs=Inputs(nTrain+1:end,:);
TestTargets=Targets(nTrain+1:end,:);
% Making Final Data Struct
data.TrainInputs=TrainInputs;
data.TrainTargets=TrainTargets;
data.TestInputs=TestInputs;
data.TestTargets=TestTargets;

%% Basic Fuzzy Model Creation 
% Number of Clusters in FCM
ClusNum=2;
%
% Creating FIS
fis=GenerateFuzzy(data,ClusNum);

%% Tarining Cultural Algorithm
CulturalAlgorithmFis = CulturalFCN(fis,data); 

%% Train Output Extraction
TrTar=data.TrainTargets;
TrainOutputs=evalfis(data.TrainInputs,CulturalAlgorithmFis);
% Train calculation
Errors=data.TrainTargets-TrainOutputs; 
r0 = -1 ;
r1 = +1 ;
range = max(Errors) - min(Errors);
Errors = (Errors - min(Errors)) / range;
range2 = r1-r0;
Errors = (Errors * range2) + r0;
MSE=mean(Errors.^2);
RMSE=sqrt(MSE);  
error_mean=mean(Errors);
error_std=std(Errors);
%% Results
% Basic Image Quantization
seg_I = imquantize(I,Data);
RGB = label2rgb(seg_I); 
% Cultural Algorithm Image Quantization
TrainOutputs(thresholdlvl)=TrainOutputs(end)+1;
TrainOutputs=sort(TrainOutputs);
seg_I2 = imquantize(I,TrainOutputs);
RGB2 = label2rgb(seg_I2); 
% Plot Results
figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,2,1)
subimage(I); title('Original Eva');
subplot(2,2,2)
subimage(RGB);title('Basic Quantization');
subplot(2,2,3)
subimage(RGB2);title('Cultural Algorithm Quantization');
subplot(2,2,4)
imhist(rgb2gray(RGB2));title('Cultural Algorithm Image Histogram');

%% Cultural Algorithm Image Quantization Performance Statistics
fprintf('Cultural Algorithm MSE Is =  %0.4f.\n',MSE)
fprintf('Cultural Algorithm RMSE Is =  %0.4f.\n',RMSE)
fprintf('Cultural Algorithm Train Error Mean Is =  %0.4f.\n',error_mean)
fprintf('Cultural Algorithm Train Error STD Is =  %0.4f.\n',error_std)
fprintf('Basic VS Cultural Algorithm Error Is =  %0.4f.\n',mse(rgb2gray(RGB),rgb2gray(RGB2)))
fprintf('Basic VS Cultural Algorithm PSNR Is =  %0.4f.\n',psnr(rgb2gray(RGB),rgb2gray(RGB2)))




