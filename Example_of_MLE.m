clear;clc;close all

%% For a univariate random variable
rng default  % For reproducibility
data = normrnd(0,1,[100,1]);  % Generate 100 random numbers from N(0,1)

phat = mle(data, 'distribution', 'normal');

negloglik = @(param) -sum(log(normpdf(data,param(1),param(2))));
start = [0,1];  % Starting values for the parameters
phat_hardcoded = fminunc(negloglik, start);

%% For a multivariate random variable

% True parameters

mu_true = [0, 0, 0];
Sigma_true = [1, 0.5, 0.3; 0.5, 1, 0.2; 0.3, 0.2, 1];

% Generate data
X = mvnrnd(mu_true, Sigma_true, 1000);

% Estimate parameters
[mu_hat, Sigma_hat] = MLE_MultivariateNormal(X);

% Display true and estimated parameters
disp('True mean:');
disp(mu_true);
disp('Estimated mean:');
disp(mu_hat);
disp('True covariance:');
disp(Sigma_true);
disp('Estimated covariance:');
disp(Sigma_hat);
%% MAP
% Set the dimensions

d = 10; % Number of features
n = 128; % Number of samples

% Generate random data for X
X = randn(d, n);

% Assume X is your d-by-128 dimensional matrix
[d, n] = size(X);

% Compute the sample mean (mu) and covariance (Sigma) for MLE
mu_mle = mean(X, 2);
Sigma_mle = cov(X');

% Assume we have a prior Gaussian distribution for the mean
mu_prior = zeros(d, 1); % Prior mean is a zero vector
Sigma_prior = eye(d); % Prior covariance is an identity matrix

% Compute the MAP estimate of the mean
mu_map = inv(inv(Sigma_prior) + n * Sigma_mle) * ...
             (Sigma_prior \ mu_prior + n * Sigma_mle * mu_mle);

% Now mu_mle and mu_map are the estimates of the mean under MLE and MAP, respectively

%% 

function [mu_hat, Sigma_hat] = MLE_MultivariateNormal(X)

    % X is a matrix where each row is a data point
    % Initialize mu and Sigma
    mu_init = mean(X);
    Sigma_init = cov(X);
    
    % Define the negative log likelihood function
    negLogLikelihood = @(theta) negLogLikelihoodMultivariateNormal(X, ...
        theta(1:size(X,2)), reshape(theta(size(X,2)+1:end), size(X,2), ...
                                                            size(X,2)));
    
    % Flatten mu and Sigma into a single vector
    theta_init = [mu_init, Sigma_init(:)'];
    
    % Use fminunc to minimize the negative log likelihood
    options = optimoptions(@fminunc, 'Algorithm', 'quasi-newton');
    theta_hat = fminunc(negLogLikelihood, theta_init, options);
    
    % Extract mu_hat and Sigma_hat from theta_hat
    mu_hat = theta_hat(1:size(X,2));
    Sigma_hat = reshape(theta_hat(size(X,2)+1:end), size(X,2), size(X,2));
end

function nll = negLogLikelihoodMultivariateNormal(X, mu, Sigma)

    % X is a matrix where each row is a data point
    % mu is the mean vector
    % Sigma is the covariance matrix

    % Number of dimensions
    d = size(X, 2);

    % Difference between data and mean
    diff = bsxfun(@minus, X, mu);

    % Log likelihood for each data point
    logLikelihood = -0.5 * sum((diff / Sigma) .* diff, 2) - ...
                                (d/2) * log(2*pi) - 0.5 * log(det(Sigma));

    % Negative log likelihood
    nll = -sum(logLikelihood);
end

