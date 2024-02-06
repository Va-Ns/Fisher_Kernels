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
numModels = 128;
logLikelihoods_SIFT = zeros(1, numModels);
Log_Likelihoods_SIFT = cell(1, numModels);
logLikelihoods_RGB = zeros(1, numModels);
Log_Likelihoods_RGB = cell(1, numModels);

AICs = zeros(1, numModels);
BICs = zeros(1, numModels);

Responsibilities_SIFT = cell(1, numModels);
Responsibilities_RGB = cell(1, numModels);

%% Για τα RGB δεδομένα
tic
for i = 1 : numModels

    fprintf("Number of cluster:%d \n",i)

    GMMs = sEM(FeatureMatrix.Reduced_RGB_Features_Matrix, i,"Alpha",0.5); 
    logLikelihoods_RGB(i) = GMMs.NegLogLikelihood;
    Log_Likelihoods_RGB{i} = GMMs.Log_Likelihood;

    fprintf(" >> Negative Log-Likelihood:%e\n ",logLikelihoods_RGB(i))   
    
end
sEM_RGB_time = toc

plot(logLikelihoods_RGB,'o','LineWidth', 2, 'MarkerSize',10, ...
                                                    'MarkerFaceColor', 'b')
grid on
title("Negative Log-Likelihood over Number of Clusters for RGB features")
xlabel("Number of Clusters")
ylabel("Negative Log-Likelihood")

%% Για τα SIFT δεδομένα

% Calculate the index for one third of the data
oneFourthIndex = floor(size(FeatureMatrix.Reduced_SIFT_Features_Matrix, 1) / 4);
SIFT_data = gpuArray(FeatureMatrix.Reduced_SIFT_Features_Matrix(1:oneFourthIndex,:));
tic
for i = 1 : numModels

    fprintf("Number of cluster:%d \n",i)

    GMMs = sEM(SIFT_data,i,"Alpha",0.5); 
    logLikelihoods_SIFT(i) = GMMs.NegLogLikelihood;
    Log_Likelihoods_SIFT{i} = GMMs.Log_Likelihood;

    fprintf(" >> Negative Log-Likelihood:%e\n ",logLikelihoods_SIFT(i))   
    
end
sEM_SIFT_time = toc

plot(logLikelihoods_SIFT,'o','LineWidth', 2, 'MarkerSize',10, ...
                                                    'MarkerFaceColor', 'b')
grid on
title("Negative Log-Likelihood over Number of Clusters for SIFT features")
xlabel("Number of Clusters")
ylabel("Negative Log-Likelihood")
%% Calculate the statistics for the SIFT and RGB features
RGB_data = gpuArray(FeatureMatrix.Reduced_RGB_Features_Matrix);

clear FeatureMatrix

GMM_Params = CalculateParamsNV(SIFT_data,RGB_data, ...
                                   Log_Likelihoods_SIFT,Log_Likelihoods_RGB);


%% Data Generation
% 
% % Get the number of clusters and features
% numClusters = size(GMM_Params(end).SIFT_Sigmas, 1);
% numFeatures = size(GMM_Params(end).SIFT_Sigmas, 2);
% 
% % Initialize the covariance matrices
% covMatrices = zeros(1, numFeatures, numClusters);
% 
% % Loop over each cluster
% for i = 1:numClusters
%     % Create a square diagonal matrix from the covariance vector
%     covMatrices(1, :, i) = GMM_Params(end).SIFT_Sigmas(i, :);
% end
% 
% % Create a gmdistribution object
% gmModel = gmdistribution(GMM_Params(end).SIFT_mus, covMatrices, GMM_Params(end).SIFT_weights);
% Y = random(gmModel, size(SIFT_data,1));

%% Create the gradient vectors

Fisher_Kernel = gradientVectorsNV(GMM_Params,Log_Likelihoods_SIFT,Log_Likelihoods_RGB);

%% Calculate AIC and BIC

% nParam = size(data, 2) + numClusters - 1 + numClusters * ...
%     size(data, 2);
% 
% AIC = 2 * nParam - 2 * bestLogLikelihood;
% BIC = nParam * log(numPoints) - 2 * bestLogLikelihood;