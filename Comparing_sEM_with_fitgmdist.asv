clear;clc
% Step 1: Generate Data
data = randn(1000000, 128);

% Number of clusters
numClusters = 100;

% Step 2: Run sEM.m
tic;
GMM = sEM(data, numClusters);
time_sEM = toc;
fprintf('\n sEM execution time: %f seconds\n', time_sEM);

% Step 3: Run fitgmdist
tic;
GMModel = fitgmdist(data, numClusters, 'Options', statset('MaxIter',100));
time_fitgmdist = toc;
fprintf('fitgmdist execution time: %f seconds\n', time_fitgmdist);