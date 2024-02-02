clear;clc;close all
%%
delete(gcp('nocreate'))
maxWorkers = maxNumCompThreads;
disp("Maximum number of workers: " + maxWorkers);
pool=parpool(maxWorkers/2+2);
%% Load the YOLOv4 Object Detector
yolo = yolov4ObjectDetector("csp-darknet53-coco");

%% Load the data 
filelocation = uigetdir;
imds = imageDatastore(filelocation);

%% Assign Labels
tic
new_imds = assignLabels(imds,yolo);
clear yolo
%% Feature Extraction
tic
features = extractImageFeatures(new_imds);
toc

%% Create a struct that contains the vertically concatenated features 

% Concatenate each feature category into a single matrix and put it into a
% structure
FeatureMatrix.SIFT_Features_Matrix = vertcat(features(:).SIFT_Features);
FeatureMatrix.RGB_Features_Matrix = vertcat(features(:).RGBfeatures);
FeatureMatrix.Reduced_SIFT_Features_Matrix = vertcat(features(:).reduced_SIFT_features);
FeatureMatrix.Reduced_RGB_Features_Matrix = vertcat(features(:).reduced_RGB_features);
preprocessing_time = toc

%% Gaussian Mixture Model

%step = 10;
numModels = 200;
logLikelihoods = zeros(1, numModels);
AICs = zeros(1, numModels);
BICs = zeros(1, numModels);
GMMs = cell(1, numModels);


for i = 1 : numModels

    fprintf("Number of cluster:%d ",i)
    GMMs{i} = sEM(FeatureMatrix.Reduced_SIFT_Features_Matrix, i); 
    logLikelihoods(i) = GMMs{i}.NegLogLikelihood;
    AICs(i) = GMMs{i}.AIC;
    BICs(i) = GMMs{i}.BIC;
    fprintf(" >> Negative Log-Likelihood:%e\n ",logLikelihoods(i))   
    
end

plot(-logLikelihoods,'.','LineWidth', 2, 'MarkerSize',10, ...
                                                    'MarkerFaceColor', 'b')
title("Negative Log-Likelihood over Number of Clusters")
xlabel("Number of Clusters")
ylabel("Negative Log-Likelihood")