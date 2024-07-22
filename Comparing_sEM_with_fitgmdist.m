clear;clc
% Step 1: Generate Data
data = randn(1000000, 2);

% Number of clusters
numClusters = 100;

% Step 2: Run sEM.m
tic;
GMM = sEM(data, numClusters);
time_sEM = toc;
fprintf('\n sEM execution time: %f seconds\n', time_sEM);

% Step 3: Run fitgmdist
tic;
GMModel = GMM_NV(data, numClusters,"MaxIterations",100);
time_GMM = toc;
fprintf('GMM execution time: %f seconds\n', time_GMM);

%%

