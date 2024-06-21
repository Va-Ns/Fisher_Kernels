clear;clc;close all
rng(1)
%%
delete(gcp('nocreate'))
maxWorkers = maxNumCompThreads;
disp("Maximum number of workers: " + maxWorkers);
pool=parpool(maxWorkers/4);
%% Load the YOLOv4 Object Detector
yolo = yolov4ObjectDetector("csp-darknet53-coco");

%% Load the data 
filelocation = uigetdir;
imds = imageDatastore(filelocation);

%% Assign Labels
tic
new_imds = assignLabels(imds,yolo);
clear yolo

%% Partition the data into training and testing sets

% splitDatastore = splitEachLabel(new_imds,1/4);
newlabels = countEachLabel(new_imds);

[Trainds,Testds] = splitTheDatastore(new_imds,newlabels);

%% Feature Extraction
tic
[Training_features,removedTrainingIndices] = extractImageFeatures(Trainds);
[Testing_features,removedTestingIndices]= extractImageFeatures(Testds);
toc

%% Create a struct that contains the vertically concatenated features 

% Concatenate each feature category into a single matrix and put it into a
% structure

tic
% FeatureMatrix.SIFT_Features_Matrix = vertcat(features(:).SIFT_Features);
% FeatureMatrix.RGB_Features_Matrix = vertcat(features(:).RGBfeatures);
Training_FeatureMatrix.Reduced_SIFT_Features_Matrix = ...
                       vertcat(Training_features(:).reduced_SIFT_features);
Training_FeatureMatrix.Reduced_RGB_Features_Matrix = ...
                        vertcat(Training_features(:).reduced_RGB_features);
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

% Για τα RGB δεδομένα
tic
for i = 1 : numModels

    fprintf("Number of cluster:%d \n",i)

    GMMs = sEM(Training_FeatureMatrix.Reduced_RGB_Features_Matrix, i,"Alpha",0.5,"BatchSize",100); 
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

% Για τα SIFT δεδομένα

% Calculate the index for one fourth of the data
oneFourthIndex = floor(size(Training_FeatureMatrix.Reduced_SIFT_Features_Matrix, 1) / 4);

Training_SIFT_data = gpuArray(Training_FeatureMatrix.Reduced_SIFT_Features_Matrix(1:oneFourthIndex,:));
SIFT_data = gpuArray(Training_FeatureMatrix.Reduced_SIFT_Features_Matrix(1:oneFourthIndex,:));

tic
for i = 1 : numModels

    fprintf("Number of cluster:%d \n",i)

    GMMs = sEM(Training_SIFT_data,i,"Alpha",0.5); 
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

Training_RGB_data = gpuArray(Training_FeatureMatrix.Reduced_RGB_Features_Matrix);
RGB_data = gpuArray(Training_FeatureMatrix.Reduced_RGB_Features_Matrix);

GMM_Params = CalculateParamsNV(Training_SIFT_data,Training_RGB_data, ...
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
fprintf("Encoding the Training Features\n")
Total_Training_Fisher_Kernel = FisherEncodingNV(GMM_Params,Training_features);
fprintf("Encoding the Training Features\n")
Total_Testing_Fisher_Kernel  = FisherEncodingNV(GMM_Params,Testing_features);

%% It's classification time

% Remove the corresponding indices from the labels 
mask = true(1, numel(Trainds.Files));
mask(removedTrainingIndices) = false;

new_Trainds = subset(Trainds, mask);

mask = true(1, numel(Testds.Files));
mask(removedTestingIndices) = false;

new_Testds = subset(Testds, mask);

t = templateSVM('SaveSupportVectors',true,'Type','classification');
[Model1,HyperparameterOptimizationResults] = fitcecoc(Total_Training_Fisher_Kernel, ...
    [new_Trainds.Labels;new_Trainds.Labels],"Learners",t,"Coding", "onevsall", ...
    'OptimizeHyperparameters',{'BoxConstraint','KernelScale'}, ...
    'HyperparameterOptimizationOptions',struct('Holdout',0.1,"UseParallel",true));

Mdl = Model1.Trained{1};
[predictedLabels, scores]= predict(Model1,Total_Testing_Fisher_Kernel);

% Αξιολόγηση
% new_Testing_Labels = [new_Testds.Labels;new_Testds.Labels];
% new_Testing_Labels(removedTestingIndices,:) = [];
confusionMatrix = confusionmat(new_Testing_Labels,predictedLabels);

% Υπολογισμός ακρίβειας
accuracy = sum(diag(confusionMatrix)) / sum(confusionMatrix(:));

% Υπολογισμός Ακρίβειας, Recall και F1-score για την κάθε κλάση
numClasses = size(confusionMatrix, 1);
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
f1Score = zeros(numClasses, 1);
for j = 1:numClasses
    precision(j) = confusionMatrix(j,j) / sum(confusionMatrix(:,j));
    recall(j) = confusionMatrix(j,j) / sum(confusionMatrix(j,:));
    f1Score(j) = 2 * (precision(j) * recall(j)) / (precision(j) + ...
                                                                recall(j));
end

%% Calculate AIC and BIC

% nParam = size(data, 2) + numClusters - 1 + numClusters * ...
%     size(data, 2);
% 
% AIC = 2 * nParam - 2 * bestLogLikelihood;
% BIC = nParam * log(numPoints) - 2 * bestLogLikelihood;