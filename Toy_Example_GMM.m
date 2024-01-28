clear;clc;close all
% Define the number of data points and the number of dimensions
numPoints = 1e6; % 1 million points
numDims = 10; % 10 dimensions

% Define the number of clusters and their parameters
numClusters = 100;
mus = randn(numClusters, numDims); % Random means
Sigmas = repmat(eye(numDims), 1, 1, numClusters); % Identity covariance matrices

% Preallocate the data array
data = zeros(numPoints, numDims);

% Generate data
for i = 1:numClusters
    % Determine the number of points for this cluster
    numPointsCluster = round(numPoints / numClusters);
    
    % Generate multivariate Gaussian data
    data((i-1)*numPointsCluster+1:i*numPointsCluster, :) = mvnrnd(mus(i, :), Sigmas(:, :, i), numPointsCluster);
end

% Convert data to a GPU array
data = gpuArray(data);

[negLogLikelihoods, weights, mus, Sigmas] = EM_Algorithm(data, 4, numDims, 100000);

% Classify a new data point
newData = gpuArray(randn(1, numDims));
cluster = toy_classify(newData, weights, mus, Sigmas,numClusters,numPoints);

% Print the cluster assignment
fprintf('The new data point was assigned to cluster %d.\n', cluster);

