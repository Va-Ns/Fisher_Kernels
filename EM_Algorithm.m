function [negLogLikelihoods, weights, mus, Sigmas] = EM_Algorithm(data, numClusters, numDims,maxIterations,numPoints)

    % Define the initial parameters
    weights = ones(1, numClusters) / numClusters; % Equal weights
    mus = gpuArray(randn(numClusters, numDims)); % Random means
    Sigmas = gpuArray(ones(numClusters, numDims)); % Unit variances

    % Define the tolerance for convergence
    tol = 1e-6;

    % Initialize the log-likelihood
    logLikelihoodOld = -Inf;

    % Initialize the vector to store the negative log-likelihood at each iteration
    negLogLikelihoods = [];

    % Vectorized Gaussian function
    Gaussian = @(X, mu, Sigma) (1 / sqrt((2*pi*(Sigma^2)))) * ...
                                exp(-((X-mu).^2/(2*(Sigma)^2)));

    for iteration = 1:maxIterations
        % E-step: Compute the responsibilities using the current parameters
        r_nk = gpuArray.zeros(numPoints, numClusters);
        for dim = 1:numDims
            gaussians = arrayfun(Gaussian, bsxfun(@minus, data(:, dim), mus(:, dim)'), ...
                                           bsxfun(@minus, mus(:, dim)', data(:, dim)), ...
                                           bsxfun(@times, sqrt(max(Sigmas(:, dim)', 1e-6)), ones(numPoints, 1)));
            r_nk = r_nk + log(max(weights, 1e-6)) + log(max(gaussians, 1e-6));
        end
        r_nk = exp(r_nk - max(r_nk, [], 2));
        r_nk = r_nk ./ sum(r_nk, 2);

        % M-step: Update the parameters using the current responsibilities
        N_k = sum(r_nk, 1);
        weights = max(N_k / numPoints, 1e-6);
        for i = 1:numClusters
            mus(i, :) = r_nk(:, i)' * data / max(N_k(i), 1e-6);
            Sigmas(i, :) = r_nk(:, i)' * (data - mus(i, :)).^2 / max(N_k(i), 1e-6);
        end

        % Compute the log-likelihood
        logLikelihood = 0;
        for dim = 1:numDims
            gaussians = arrayfun(Gaussian, bsxfun(@minus, data(:, dim), mus(:, dim)'), ...
                                           bsxfun(@minus, mus(:, dim)', data(:, dim)), ...
                                           bsxfun(@times, sqrt(max(Sigmas(:, dim)', 1e-6)), ones(numPoints, 1)));
            logLikelihood = logLikelihood + weights * sum(log(max(gaussians, 1e-6)), 1)';
        end

        % Store the negative log-likelihood
        negLogLikelihoods = [negLogLikelihoods, -gather(logLikelihood)];

        % Check for convergence
        if abs(logLikelihood - logLikelihoodOld) < tol
            break;
        end

        logLikelihoodOld = logLikelihood;
    end

    % Plot the negative log-likelihood over iterations
    figure;
    plot(negLogLikelihoods);
    xlabel('Iteration');
    ylabel('Negative Log-Likelihood');
end