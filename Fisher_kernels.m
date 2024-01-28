clear;clc;close all

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

GMM = EM_Algorithm(FeatureMatrix.Reduced_SIFT_Features_Matrix,10,"maxIterations",10,"numReplicates",10);

options = statset('MaxIter',10,'Display','iter');
SIFTGMMmodel = fitgmdist(FeatureMatrix.Reduced_SIFT_Features_Matrix,10, ...
                         "CovarianceType","diagonal","Options",options,"Replicates",10);

