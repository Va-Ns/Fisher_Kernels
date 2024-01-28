function GMM = EM_Algorithm(data, numClusters, Options)
    
    arguments 

        data (:,:) double {mustBeReal, mustBeFinite}
        numClusters (1,1) double {mustBeInteger, mustBePositive}
        Options.maxIterations (1,1) double {mustBeInteger, mustBePositive} = 100;
        Options.numReplicates (1,1) double {mustBeInteger, mustBePositive} = 10;
        Options.tolerance (1,1) double {mustBeReal, mustBeFinite} = 1e-8;

    end


   [numPoints,numDims] = size(FeatureMatrix.Reduced_SIFT_Features_Matrix);

    % Define the tolerance for convergence
    tol = Options.tolerance;

    % Initialize the log-likelihood
    logLikelihoodOld = -Inf;

    bestLogLikelihood = Inf;
    bestLogLikelihoodOld = Inf;
    bestReplicate = 0;

    % Preallocate log_lh
    Log_Likelihood = gpuArray.zeros(numPoints, numClusters);

    % Compute constant term
    constTerm = numDims*log(2*pi)/2;

    for replicate = 1:numReplicates
        
        % Define the initial parameters
        weights = ones(1, numClusters) / numClusters; % Equal weights
        mus = gpuArray(randn(numClusters, numDims)); % Random means
        Sigmas = gpuArray(ones(numClusters, numDims)); % Unit variances

        for iteration = 1:maxIterations
            
            %% E-step: Compute the responsibilities using the current parameters
            Log_Likelihood(:) = 0; % Reset log_lh
            for j = 1:numClusters

                
                log_prior = log(weights(j));
                %^^^^^^^^^^^^^^^^^^^^^^^^^^^
                % This line calculates the logarithm of the prior probability 
                % of the j-th cluster. The prior probability is represented 
                % by the weight of the cluster,which indicates how likely 
                % it is that a randomly chosen data point belongs to this 
                % cluster.

                
                logDetSigma = sum(log(Sigmas(j, :)));
                %^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                % This line calculates the logarithm of the determinant of 
                % the covariance matrix Sigmas for the j-th cluster. Since 
                % the covariance matrix is diagonal in this case, the 
                % determinant is simply the product of the diagonal elements,
                % and the logarithm of the determinant is the sum of the 
                % logarithms of the diagonal elements.
         
                L = sqrt(Sigmas(j, :));
                %^^^^^^^^^^^^^^^^^^^^^^
                % This line calculates the square root of each element in 
                % the j-th row of Sigmas, which represents the standard 
                % deviations of the j-th cluster.

                Log_Likelihood(:,j) = sum(bsxfun(@rdivide, bsxfun(@minus, ...
                                                data, mus(j,:)), L).^2, 2);
                %^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                % This line calculates the squared Mahalanobis distance 
                % from each data point to the mean of the j-th cluster, 
                % divided by the variance. This is done using element-wise 
                % operations and broadcasting, which makes the code more 
                % efficient and easier to read.

                Log_Likelihood(:,j) = -0.5*(Log_Likelihood(:,j) + logDetSigma);
                %^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                % The log-likelihood is being calculated.

                Log_Likelihood(:,j) = Log_Likelihood(:,j) + log_prior - constTerm;
                %^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                % This line adds the log-prior and subtracts a constant 
                % term from the log-likelihood. The constant term does not 
                % depend on the cluster, so it does not affect which 
                % cluster has the maximum log-likelihood, but it ensures 
                % that the log-likelihoods are properly normalized.
            end

            MaxLogLikelihood = max(Log_Likelihood,[],2);
            responsibilities = exp(Log_Likelihood-MaxLogLikelihood); % To avoid numerical 
                                                                     % underflow,normalize 
                                                                     % the responsibilities 
                                                                     % and exponentiate. 
            
            %  This line computes the sum of the responsibilities for each 
            % data point across all clusters. The result is a column vector
            % where each element is the sum of the responsibilities for a 
            % data point.
            Density = sum(responsibilities,2);
            Logpdf = log(Density) + MaxLogLikelihood; 
            % ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            % The addition of the maximum log-likelihood to the log of the 
            % density is a common technique used to improve numerical 
            % stability when working with very small (or very large) numbers,
            % which is often the case when dealing with probabilities and 
            % likelihoods.
            logLikelihood = sum(Logpdf);
            % ^^^^^^^^^^^^^^^^^^^^^^^^^^
            % Τhis line computes the total log-likelihood, which is the sum
            % of the log-pdfs for all data points.
            responsibilities = responsibilities./Density; % Νormalize Responsibilities

            %% M-step: Update the parameters using the current responsibilities
            
            for j = 1:numClusters
                % For each cluster

                Responsibilities_j = responsibilities(:,j)';
                % ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                % Extract the responsibilities of all data points for the 
                % current cluster j. The responsibilities represent the 
                % probability that each data point belongs to the current 
                % cluster, given the current parameter estimates.

                Nonzero_idx = Responsibilities_j > 0;
                %^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                % Create a logical index vector that identifies the data 
                % points for which the responsibility towards the current 
                % cluster j is greater than zero.
                mus(j,:) = Responsibilities_j * data / sum(Responsibilities_j);
                %^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                % Update the mean vector mus for the current cluster j. 
                % The new mean is the weighted average of all data points, 
                % where the weights are the responsibilities. This is the 
                % expected value of the data given the current responsibilities
                Centered_Data = data(Nonzero_idx,:) - mus(j,:);
                %^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                % Computes the centered data by subtracting the new mean 
                % mus(j,:) from the data points that have non-zero 
                % responsibility towards the current cluster j.
                Sigmas(j,:) = Responsibilities_j(Nonzero_idx) *... 
                              (Centered_Data.^2) / sum(Responsibilities_j(Nonzero_idx)) +...
                               1e-6;
                %^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 
                % This line updates the variance vector Sigmas for the 
                % current cluster j. The new variance is the weighted 
                % average of the squared centered data, where the weights 
                % are the responsibilities. The 1e-6 term is added for 
                % numerical stability, to prevent division by zero or 
                % taking the square root of a negative number in subsequent
                % computations.
                weights(j) = sum(Responsibilities_j) / numPoints;
                %^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                % This line updates the weight for the current cluster j. 
                % The new weight is the average responsibility of the data
                % points towards the current cluster. This represents the 
                % estimated probability that a randomly chosen data point 
                % belongs to the current cluster.
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
    nParam = size(data, 2) + numClusters - 1 + numClusters * size(data, 2); % Number of parameters
    
    % Compute AIC and BIC
    AIC = 2*nParam + 2 * bestLogLikelihood;
    BIC = nParam*log(numPoints) + 2 * bestLogLikelihood;

    % Create a structure that contains the parameters, log-likelihood, AIC, and BIC
    GMM = struct('weights', weights, 'mus', mus, 'Sigmas', Sigmas, ...
                 'logLikelihood', bestLogLikelihood, 'AIC', AIC, 'BIC', BIC);
end