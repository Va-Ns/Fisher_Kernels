function [bestNegLogLikelihood, weights, mus, Sigmas, AIC, BIC] = EM_Algorithm2(data, numClusters, numDims, maxIterations, numPoints, numReplicates)

    % Define the tolerance for convergence
    tol = 1e-8;

    % Initialize the log-likelihood
    logLikelihoodOld = -Inf;

    % Initialize the best negative log-likelihood to Inf
    bestNegLogLikelihood = Inf;

    bestLogLikelihood = Inf;
    bestLogLikelihoodOld = Inf;
    bestReplicate = 0;

    % Preallocate log_lh
    log_lh = gpuArray.zeros(numPoints, numClusters);

    % Compute constant term
    constTerm = numDims*log(2*pi)/2;

    for replicate = 1:numReplicates
        
        % Define the initial parameters
        weights = ones(1, numClusters) / numClusters; % Equal weights
        mus = gpuArray(randn(numClusters, numDims)); % Random means
        Sigmas = gpuArray(ones(numClusters, numDims)); % Unit variances

        for iteration = 1:maxIterations
            
            %% E-step: Compute the responsibilities using the current parameters
            log_lh(:) = 0; % Reset log_lh
            for j = 1:numClusters
                log_prior = log(weights(j));
                logDetSigma = sum(log(Sigmas(j, :)));
                L = sqrt(Sigmas(j, :));
                log_lh(:,j) = sum(bsxfun(@rdivide, bsxfun(@minus, data, mus(j,:)), L).^2, 2);
                log_lh(:,j) = -0.5*(log_lh(:,j) + logDetSigma);
                log_lh(:,j) = log_lh(:,j) + log_prior - constTerm;
            end

            maxll = max(log_lh,[],2);
            post = exp(log_lh-maxll);
            density = sum(post,2);
            logpdf = log(density) + maxll;
            logLikelihood = sum(logpdf); 
            post = post./density;%normalize posteriors

            %% M-step: Update the parameters using the current responsibilities
            
            for j = 1:numClusters
                post_j = post(:,j)';
                nz_idx = post_j>0;
                mus(j,:) = post_j * data / sum(post_j);
                Xcentered = data(nz_idx,:) - mus(j,:);
                Sigmas(j,:) = post_j(nz_idx) * (Xcentered.^2) / sum(post_j(nz_idx)) + 1e-6; % Corrected line
                weights(j) = sum(post_j) / numPoints;
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
            bestReplicate = replicate;
        end
        
        % Check if the best log-likelihood has improved significantly
        if abs(bestLogLikelihood - bestLogLikelihoodOld) < tol
            break;
        end

        bestLogLikelihoodOld = bestLogLikelihood;
        
    end

    % Return the best parameters
    weights = bestWeights;
    mus = bestMus;
    Sigmas = bestSigmas;

    % Compute the AIC and BIC
    p = numClusters * (1 + numDims + numDims); % Number of parameters
    AIC = 2 * p + 2 * bestLogLikelihood;
    BIC = log(numPoints) * p + 2 * bestLogLikelihood;

    % Display the replicate at which the best negative log-likelihood was found and its value
    fprintf(['Best negative log-likelihood found to be: %e at Replicate: ' ...
        '%d\n'], bestLogLikelihood, bestReplicate);
end