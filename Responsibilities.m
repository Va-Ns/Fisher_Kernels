clear;clc;close all

% Generate a million points
X = linspace(-3, 5, 3e6);

% User-provided weights, means, and standard deviations
weights = [1/4, 1/4, 1/4, 1/4];  % Replace with user-provided weights
mus = [-4, 0, 8, 2];  % Replace with user-provided means
Sigmas = [1, 0.2, 3, 0.5];  % Replace with user-provided standard deviations

% Check that the number of weights, means, and standard deviations are the same
assert(length(weights) == length(mus) && length(mus) == length(Sigmas), 'The number of weights, means, and standard deviations must be the same.');

% Combine the means and standard deviations into one matrix
parameters = [mus' Sigmas'];

tic 
% Vectorized Gaussian function
Gaussian = @(X, mu, Sigma) (1 / sqrt((2*pi*(Sigma^2)))) * exp(-((X-mu).^2/(2*(Sigma)^2)));

% Calculate Gaussian for all combinations of X and parameters
gaussians = arrayfun(Gaussian, repmat(X', 1, size(parameters, 1)), repmat(parameters(:, 1)', length(X), 1), repmat(parameters(:, 2)', length(X), 1));

% Preallocate r_nk
r_nk = zeros(length(X), length(weights));

% Calculate r_nk
r_nk = (weights .* gaussians) ./ sum(weights .* gaussians, 2);

Respo_time = toc

%% GPU

clear;close all
gpuDevice(1); % Reset the GPU

% Generate a million points
X = gpuArray.linspace(-3, 5, 3e6);

% User-provided weights, means, and standard deviations
weights = gpuArray([1/4, 1/4, 1/4, 1/4]);  % Replace with user-provided weights
mus = gpuArray([-4, 0, 8, 2]);  % Replace with user-provided means
Sigmas = gpuArray([1, 0.2, 3, 0.5]);  % Replace with user-provided standard deviations

% Check that the number of weights, means, and standard deviations are the same
assert(length(weights) == length(mus) && length(mus) == length(Sigmas), 'The number of weights, means, and standard deviations must be the same.');

% Combine the means and standard deviations into one matrix
parameters = [mus' Sigmas'];

tic
% Vectorized Gaussian function
Gaussian = @(X, mu, Sigma) (1 / sqrt((2*pi*(Sigma^2)))) * ...
                            exp(-((X-mu).^2/(2*(Sigma)^2)));

% Calculate Gaussian for all combinations of X and parameters
gaussians = arrayfun(Gaussian, repmat(X', 1, size(parameters, 1)), ...
                               repmat(parameters(:, 1)', length(X), 1), ...
                               repmat(parameters(:, 2)', length(X), 1));

% Preallocate r_nk
r_nk = gpuArray.zeros(length(X), length(weights));

% Calculate r_nk
r_nk = (weights .* gaussians) ./ sum(weights .* gaussians, 2);
GPU_time = toc
% Gather the result back to CPU memory
r_nk = gather(r_nk);

% Compute the total responsibility of the k-th mixture component

K_th_Respo = sum(r_nk);