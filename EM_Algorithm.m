function [bestNegLogLikelihood, weights, mus, Sigmas, AIC, BIC] = EM_Algorithm(data, numClusters, numDims, maxIterations, numPoints, numReplicates)

    % Define the tolerance for convergence
    tol = 1e-8;

    % Initialize the log-likelihood
    logLikelihoodOld = -Inf;

    % Initialize the best negative log-likelihood to Inf
    bestNegLogLikelihood = Inf;

    % Vectorized Gaussian function
    Gaussian = @(X, mu, Sigma) (1 / sqrt((2*pi*(Sigma^2)))) * ...
                                exp(-((X-mu).^2/(2*(Sigma)^2)));

    bestLogLikelihood = Inf;
    bestLogLikelihoodOld = Inf;  % NEW: Initialize old best log-likelihood
    bestReplicate = 0;  % NEW: Initialize best replicate

    for replicate = 1:numReplicates
        % Define the initial parameters
        weights = ones(1, numClusters) / numClusters; % Equal weights
        mus = gpuArray(randn(numClusters, numDims)); % Random means
        Sigmas = gpuArray(ones(numClusters, numDims)); % Unit variances

        for iteration = 1:maxIterations
            
            %% E-step: Compute the responsibilities using the current parameters
            r_nk = gpuArray.zeros(numPoints, numClusters);
            for dim = 1:numDims
                gaussians = arrayfun(Gaussian, bsxfun(@minus, data(:, dim), mus(:, dim)'), ...
                                               bsxfun(@minus, mus(:, dim)', data(:, dim)), ...
                                               bsxfun(@times, sqrt(max(Sigmas(:, dim)', 1e-6)), ones(numPoints, 1)));
                r_nk = r_nk + log(max(weights, 1e-6)) + log(max(gaussians, 1e-6));
            end
            r_nk = exp(r_nk - max(r_nk, [], 2));
            r_nk = r_nk ./ sum(r_nk, 2);

            %% M-step: Update the parameters using the current responsibilities

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
                logLikelihood = logLikelihood + sum(log(max(weights .* gaussians, 1e-6)), 'all');
            end

            % Check for convergence
            if abs(logLikelihood - logLikelihoodOld) < tol
                break;
            end

            logLikelihoodOld = logLikelihood;
        end

        % Check if this replicate has a better log-likelihood
        if -logLikelihood < bestLogLikelihood
            bestLogLikelihood = -logLikelihood;
            bestWeights = weights;
            bestMus = mus;
            bestSigmas = Sigmas;
            bestReplicate = replicate;  % NEW: Update best replicate
        end
        
        % NEW: Check if the best log-likelihood has improved significantly
        if abs(bestLogLikelihood - bestLogLikelihoodOld) < tol
            break;
        end
        bestLogLikelihoodOld = bestLogLikelihood;  % NEW: Update old best log-likelihood

        % Update the best negative log-likelihood if necessary
        if -logLikelihood < bestNegLogLikelihood
            bestNegLogLikelihood = -logLikelihood;
        end
    end

    % Return the best parameters
    weights = bestWeights;
    mus = bestMus;
    Sigmas = bestSigmas;

    % Compute the AIC and BIC
    p = numClusters * (1 + numDims + numDims); % Number of parameters
    AIC = 2 * p + 2 * bestLogLikelihood;
    BIC = log(numPoints) * p + 2 * bestLogLikelihood;

    % NEW: Display the replicate at which the best negative log-likelihood was found and its value
    fprintf(['Best negative log-likelihood found to be: %e at Replicate: ' ...
        '%d\n'], bestLogLikelihood, bestReplicate);
end