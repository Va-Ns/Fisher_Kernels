clear;clc;close all
s = rng(1);

mkdir Workspace

%%
delete(gcp('nocreate'))
maxWorkers = maxNumCompThreads;
disp("Maximum number of workers: " + maxWorkers);
pool=parpool(maxWorkers/2);

%% Load the data 
filelocation = uigetdir;
imds = imageDatastore(filelocation,"IncludeSubfolders",true,"LabelSource","foldernames");

%% Partition the data into training and testing sets

Labels = countEachLabel(imds)

[Trainds,Testds] = splitTheDatastore(imds,Labels);

%% Feature Extraction
tic
[Training_features,removedTrainingIndices] = extractImageFeatures(Trainds);
[Testing_features,removedTestingIndices]= extractImageFeatures(Testds);


% Create a struct that contains the vertically concatenated training features 

% Concatenate each feature category into a single matrix and put it into a
% structure

% FeatureMatrix.SIFT_Features_Matrix = vertcat(features(:).SIFT_Features);
% FeatureMatrix.RGB_Features_Matrix = vertcat(features(:).RGBfeatures);
Training_FeatureMatrix.Reduced_SIFT_Features_Matrix = ...
                       vertcat(Training_features(:).reduced_SIFT_features);
Training_FeatureMatrix.Reduced_RGB_Features_Matrix = ...
                        vertcat(Training_features(:).reduced_RGB_features);
preprocessing_time = toc;

%% Initializations

numModels = 128;

fitGMMNeglogLikelihoods_SIFT = zeros(1, numModels);
fitGMMLog_Likelihoods_SIFT = cell(1, numModels);
fitGMMNeglogLikelihoods_RGB = zeros(1, numModels);
fitGMMLog_Likelihoods_RGB = cell(1, numModels);

GMMNeglogLikelihoods_SIFT = zeros(1, numModels);
GMMLog_Likelihoods_SIFT = cell(1, numModels);
GMMNeglogLikelihoods_RGB = zeros(1, numModels);
GMMLog_Likelihoods_RGB = cell(1, numModels);

sEMNeglogLikelihoods_SIFT = zeros(1, numModels);
sEMLog_Likelihoods_SIFT = cell(1, numModels);
sEMNeglogLikelihoods_RGB = zeros(1, numModels);
sEMLog_Likelihoods_RGB = cell(1, numModels);

Training_RGB_data = gpuArray(Training_FeatureMatrix.Reduced_RGB_Features_Matrix);

% Calculate the index for one fourth of the data
oneFourthIndex = floor(size(Training_FeatureMatrix.Reduced_SIFT_Features_Matrix, 1) / 4);

Training_SIFT_data = gpuArray(Training_FeatureMatrix.Reduced_SIFT_Features_Matrix(1:oneFourthIndex,:));

clear Training_FeatureMatrix 
%% fitgmdist Gaussian Mixture Model

% Για τα RGB δεδομένα
opts = statset('Display','final','MaxIter',1500);
regularizationValue = 1e-5; % Insert a regularization value to avoid ill-conditioned covariance.
                            % This can happen when the covariance matrices during the fitting of a 
                            % Gaussian Mixture Model (GMM) become nearly singular or not positive 
                            % definite. 
tic
for i = 1 : numModels

    fprintf("Number of cluster:%d \n",i)

    fitGMMs = fitgmdist(gather(Training_RGB_data),i,"Replicates",1,"Options",opts, ...
        "CovarianceType","diagonal","RegularizationValue",regularizationValue);
    fitGMMNeglogLikelihoods_RGB(i) = fitGMMs.NegativeLogLikelihood;

    fitGMMLog_Likelihoods_RGB{i} = -fitGMMs.NegativeLogLikelihood;

end
fitGMM_RGB_time = toc;

% Για τα SIFT δεδομένα

tic
for i = 1 : numModels

    fprintf("Number of cluster:%d \n",i)

    fitGMMs = fitgmdist(gather(Training_SIFT_data),i,"CovarianceType","diagonal","Replicates",1, ...
                                                                                    "Options",opts); 
    fitGMMNeglogLikelihoods_SIFT(i) = fitGMMs.NegativeLogLikelihood;

    fitGMMLog_Likelihoods_SIFT{i} = -fitGMMs.NegativeLogLikelihood;
   
end
fitGMM_SIFT_time = toc;


%% Gaussian Mixture Model

% Για τα RGB δεδομένα

tic
for i = 1 : numModels

    fprintf("Number of cluster:%d \n",i)

    GMMs = GMM_NV(Training_RGB_data,i,"NumReplicates",1,"MaxIterations",1500);

    GMMNeglogLikelihoods_RGB(i) = -GMMs.logLikelihood;
    GMMLog_Likelihoods_RGB{i} = GMMs.logLikelihood; 
    
end
GMM_RGB_time = toc;

% Για τα SIFT δεδομένα

tic
for i = 1 : numModels

    fprintf("Number of cluster:%d \n",i)

    GMMs = GMM_NV(Training_SIFT_data,i,"NumReplicates",1,"MaxIterations",1500); 

    GMMNeglogLikelihoods_SIFT(i) = -GMMs.logLikelihood;
    GMMLog_Likelihoods_SIFT{i} = GMMs.logLikelihood;  
    
end
GMM_SIFT_time = toc;

%% sEM

% Για τα RGB δεδομένα

for i = 1 : numModels

    fprintf("Number of cluster:%d \n",i)

    sEM_GMMs = sEM(Training_RGB_data, i,"Alpha",0.5,"BatchSize",100,"MaxIterations",1500); 
    
    sEMNeglogLikelihoods_RGB(i) = sEM_GMMs.NegLogLikelihood;
    sEMLog_Likelihoods_RGB{i} = -sEM_GMMs.NegLogLikelihood;
   
    
end
sEM_RGB_time = toc;

% Για τα SIFT δεδομένα

tic
for i = 1 : numModels

    fprintf("Number of cluster:%d \n",i)

    GMMs = sEM(Training_SIFT_data,i,"Alpha",0.5,"BatchSize",100,"MaxIterations",1500); 
    
    sEMNeglogLikelihoods_SIFT(i) = sEM_GMMs.Log_Likelihood;
    sEMLog_Likelihoods_SIFT{i} = -GMMs.Log_Likelihood;  
    
end
sEM_SIFT_time = toc;

%% Calculate the statistics for the SIFT and RGB features from the function 

% For the fitgmdist function

fitgmdist_Params = CalculateParamsNV(Training_SIFT_data,Training_RGB_data,fitGMMLog_Likelihoods_RGB, ...
                                                                        fitGMMLog_Likelihoods_SIFT);

% From the GMM function

GMM_Params = CalculateParamsNV(Training_SIFT_data,Training_RGB_data,GMMLog_Likelihoods_RGB, ...
                                                                        GMMLog_Likelihoods_SIFT);

% From the sEM function

sEM_Params = CalculateParamsNV(Training_SIFT_data,Training_RGB_data,sEMLog_Likelihoods_RGB, ...
                                                                        sEMLog_Likelihoods_SIFT);

clear Training_SIFT_data Training_RGB_data 


%% Create the gradient vectors for the function 

% For the fitgmdist function

fprintf("Encoding the Training Features\n")
fitgmdist_Total_Training_Fisher_Kernel = FisherEncodingNV(fitgmdist_Params,Training_features);
fprintf("Encoding the Training Features\n")
fitgmdist_Total_Testing_Fisher_Kernel  = FisherEncodingNV(fitgmdist_Params,Testing_features);


%For the GMM function

fprintf("Encoding the Training Features\n")
GMM_Total_Training_Fisher_Kernel = FisherEncodingNV(GMM_Params,Training_features);
fprintf("Encoding the Training Features\n")
GMM_Total_Testing_Fisher_Kernel = FisherEncodingNV(GMM_Params,Testing_features);

% For the sEM function

fprintf("Encoding the Training Features\n")
sEM_Total_Training_Fisher_Kernel = FisherEncodingNV(sEM_Params,Training_features);
fprintf("Encoding the Training Features\n")
sEM_Total_Testing_Fisher_Kernel = FisherEncodingNV(sEM_Params,Testing_features);

clear Training_features Testing_features 


%% It's classification time

% Remove the corresponding indices from the labels 
mask = true(1, numel(Trainds.Files));
mask(removedTrainingIndices) = false;

new_Trainds = subset(Trainds, mask);

mask = true(1, numel(Testds.Files));
mask(removedTestingIndices) = false;

new_Testds = subset(Testds, mask);

t = templateSVM('SaveSupportVectors',true,'Type','classification');

%% For fitgmdist

[Model1,HyperparameterOptimizationResults1] = fitcecoc(fitgmdist_Total_Training_Fisher_Kernel, ...
    [new_Trainds.Labels;new_Trainds.Labels],"Learners",t,"Coding", "onevsall", ...
    'OptimizeHyperparameters','all', 'HyperparameterOptimizationOptions', ...
    struct('Holdout',0.1,"UseParallel",true));

Mdl1 = Model1.Trained{1};
[predictedLabels1, scores1]= predict(Model1,fitgmdist_Total_Testing_Fisher_Kernel);

%% For GMM

[Model2,HyperparameterOptimizationResults2] = fitcecoc(GMM_Total_Training_Fisher_Kernel, ...
    [new_Trainds.Labels;new_Trainds.Labels],"Learners",t,"Coding", "onevsall", ...
    'OptimizeHyperparameters','all', 'HyperparameterOptimizationOptions', ...
    struct('Holdout',0.1,"UseParallel",true));

Mdl2 = Model2.Trained{1};
[predictedLabels2, scores2]= predict(Model2,GMM_Total_Testing_Fisher_Kernel);


%% For sEM

[Model3,HyperparameterOptimizationResults3] = fitcecoc(sEM_Total_Training_Fisher_Kernel, ...
    [new_Trainds.Labels;new_Trainds.Labels],"Learners",t,"Coding", "onevsall", ...
    'OptimizeHyperparameters','all', 'HyperparameterOptimizationOptions', ...
    struct('Holdout',0.1,"UseParallel",true));

Mdl3 = Model3.Trained{1};
[predictedLabels3, scores3]= predict(Model3,sEM_Total_Testing_Fisher_Kernel);


confusionMatrix1 = confusionmat(new_Testing_Labels,predictedLabels1);
confusionMatrix2 = confusionmat(new_Testing_Labels,predictedLabels2);
confusionMatrix3 = confusionmat(new_Testing_Labels,predictedLabels3);

accuracy1 = sum(diag(confusionMatrix1)) / sum(confusionMatrix1(:));
accuracy2 = sum(diag(confusionMatrix2)) / sum(confusionMatrix2(:));
accuracy3 = sum(diag(confusionMatrix3)) / sum(confusionMatrix3(:));


% Initialize variables for total times
totalTimes = struct('fitgmdist', 0, 'GMM', 0, 'sEM', 0);

% After each algorithm's execution, store its total time
totalTimes.fitgmdist = fitGMM_RGB_time + fitGMM_SIFT_time;
totalTimes.GMM = GMM_RGB_time + GMM_SIFT_time;
% Assuming sEM_RGB_time and sEM_SIFT_time are calculated similarly for sEM
totalTimes.sEM = sEM_RGB_time + sEM_SIFT_time;

% Create a table for times
timesTable = struct2table(totalTimes);

% After calculating accuracies, store them
accuracies = [accuracy1, accuracy2, accuracy3];

% Create a table for accuracies
accuraciesTable = array2table(accuracies, 'VariableNames', {'fitgmdist', 'GMM', 'sEM'});

%% Plotting the progression of the negative log likelihood per iteration per algorithm

% Plot for RGB Data
figure;
hold on;
plot(fitGMMNeglogLikelihoods_RGB, 'r-', 'DisplayName', 'fitgmdist RGB');
plot(GMMNeglogLikelihoods_RGB, 'g-', 'DisplayName', 'GMM RGB');
plot(sEMNeglogLikelihoods_RGB, 'b-', 'DisplayName', 'sEM RGB');
legend('show');
xlabel('Number of Clusters');
ylabel('Negative Log Likelihood');
title('Negative Log Likelihood for RGB Data Across Algorithms');
hold off;
% Save the figure
savefig('Workspace/NLL_RGB.fig');

% Plot for SIFT Data
figure;
hold on;
plot(fitGMMNeglogLikelihoods_SIFT, 'r-', 'DisplayName', 'fitgmdist SIFT');
plot(GMMNeglogLikelihoods_SIFT, 'g-', 'DisplayName', 'GMM SIFT');
plot(sEMNeglogLikelihoods_SIFT, 'b-', 'DisplayName', 'sEM SIFT');
legend('show');
xlabel('Number of Clusters');
ylabel('Negative Log Likelihood');
title('Negative Log Likelihood for SIFT Data Across Algorithms');
hold off;
% Save the figure
savefig('Workspace/NLL_SIFT.fig');

% Assuming timesTable and accuraciesTable are already defined as MATLAB tables
% Save the tables
writetable(timesTable, 'Workspace/timesTable.csv');
writetable(accuraciesTable, 'Workspace/accuraciesTable.csv');