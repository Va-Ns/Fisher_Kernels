clear;clc;close all

%% Load the YOLOv4 Object Detector
yolo = yolov4ObjectDetector("csp-darknet53-coco");

%% Load the data 
filelocation = uigetdir;
imds = imageDatastore(filelocation);

%% Assign Labels
new_imds = assignLabels(imds,yolo);
clear yolo
%% Feature Extraction
tic
features = extractImageFeatures(new_imds);
toc
save("Fisher_kernels.m","features")