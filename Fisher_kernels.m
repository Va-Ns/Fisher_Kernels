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
tic
for i = 1 : numModels

    fprintf("Number of cluster:%d \n",i)

    GMMs = sEM(FeatureMatrix.Reduced_SIFT_Features_Matrix(1:oneFourthIndex,:), ...
               i,"Alpha",0.5); 
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
%% Calculate the statistics for the SIFT features
SIFT_data = gpuArray(FeatureMatrix.Reduced_SIFT_Features_Matrix(1:oneFourthIndex,:));

for j = 1:numModels
        
    Corr_LogLikelihood = gather(Log_Likelihoods_SIFT{j});

    get_numCluster = size(Corr_LogLikelihood,2);

    MaxCorrLogLikelihood = max(Corr_LogLikelihood,[],2);
    Responsibilities = gpuArray(exp(Corr_LogLikelihood - MaxCorrLogLikelihood));

    for k = 1 : get_numCluster

        Responsibilities_k = Responsibilities(:,k)';
    
        Nonzero_idx = Responsibilities_k > 0;
    
        mus(k,:) = Responsibilities_k * data...
                                                    / sum(Responsibilities_k);
    
        Centered_Data = data(Nonzero_idx,:) - mus(k,:);
    
        Sigmas(k,:) = Responsibilities_k(Nonzero_idx) *...
            (Centered_Data.^2) / sum(Responsibilities_k(Nonzero_idx)) + 1e-6;
    
        weights(k) = sum(Responsibilities_k) / size(data,1);

    end

    GMM_Params(j).mus = gather(mus);
    GMM_Params(j).Sigmas = gather(Sigmas);
    GMM_Params(j).weights = gather(weights);
end

%% Calculate AIC and BIC

nParam = size(data, 2) + numClusters - 1 + numClusters * ...
    size(data, 2);

AIC = 2 * nParam - 2 * bestLogLikelihood;
BIC = nParam * log(numPoints) - 2 * bestLogLikelihood;