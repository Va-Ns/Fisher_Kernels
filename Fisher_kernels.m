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

k = 1:128;
nK = numel(k);

Sigma = 'diagonal';
nSigma = numel(Sigma);

SharedCovariance = {true,false};
SCtext = {'true','false'};
nSC = numel(SharedCovariance);

RegularizationValue = 0.01;
options = statset('MaxIter',1000);
X = FeatureMatrix.Reduced_SIFT_Features_Matrix;

% Fit all models in parallel
for m = 1:nSC
    for i = 1:nK
        gm{i,m} = fitgmdist(X,k(i),...
                            'CovarianceType',Sigma,...
                            'SharedCovariance',SharedCovariance{m},...
                            'RegularizationValue',RegularizationValue,...
                            'Options',options);
        aic(i,m) = gm{i,m}.AIC;
        bic(i,m) = gm{i,m}.BIC;
        converged(i,m) = gm{i,m}.Converged;
    end
end

allConverge = (sum(converged(:)) == nK*nSigma*nSC);

figure
bar(reshape(aic,nK,nSigma*nSC))
title('AIC For Various $k$','Interpreter','latex')
xlabel('$k$','Interpreter','Latex')
ylabel('AIC')
legend({'Diagonal-shared','Diagonal-unshared'})

figure
bar(reshape(bic,nK,nSigma*nSC))
title('BIC For Various $k$','Interpreter','latex')
xlabel('$c$','Interpreter','Latex')
ylabel('BIC')
legend({'Diagonal-shared','Diagonal-unshared'})   