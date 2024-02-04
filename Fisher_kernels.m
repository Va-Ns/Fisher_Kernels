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
Responsibilities = cell(1, numModels);

tic
for i = 1 : numModels

    fprintf("Number of cluster:%d \n",i)
    GMMs = sEM(FeatureMatrix.Reduced_SIFT_Features_Matrix, i,"Alpha",0.5); 
    logLikelihoods(i) = GMMs.NegLogLikelihood;
    AICs(i) = GMMs.AIC;
    BICs(i) = GMMs.BIC;
    fprintf(" >> Negative Log-Likelihood:%e\n ",logLikelihoods(i))   
    
end
sEM_time = toc

plot(-logLikelihoods,'o','LineWidth', 2, 'MarkerSize',10, ...
                                                    'MarkerFaceColor', 'b')
title("Negative Log-Likelihood over Number of Clusters")
xlabel("Number of Clusters")
ylabel("Negative Log-Likelihood")

%% Calculate the statistics

for j = 1:numClusters

    Responsibilities_j = Responsibilities(:,j)';

    Nonzero_idx = Responsibilities_j > 0;

    mus(j,:) = Responsibilities_j * data / sum(Responsibilities_j);

    Centered_Data = data(Nonzero_idx,:) - mus(j,:);

    Sigmas(j,:) = Responsibilities_j(Nonzero_idx) *...
        (Centered_Data.^2) / sum(Responsibilities_j(Nonzero_idx)) + 1e-6;

    weights(j) = sum(Responsibilities_j) / numPoints;

end