function GMM = sEM(data,numClusters,Options)
    
    arguments 

        data (:,:) double {mustBeReal, mustBeFinite}
        numClusters (1,1) double {mustBeInteger, mustBePositive,...
                                  mustBeNonempty,mustBeNonzero,mustBeNonmissing}
        Options.MaxIterations (1,1) double {mustBeInteger, mustBePositive} = 1000;
        Options.Tolerance (1,1) double {mustBeReal, mustBeFinite} = 1e-8;
        Options.Alpha     (1,1) double  {mustBeInRange(Options.Alpha,0.5,1),...
                                        mustBePositive, mustBeFloat} = 0.7
        Options.BatchSize     (1,1) double {mustBeInRange(Options.BatchSize, ...
                                            1,1000000),mustBePositive,... 
                                            mustBeInteger} = 10000
                                        
    end

    [numPoints, dimFeatures] = size(data);
       
    k = 0;
    numIterations = Options.MaxIterations;
    a = Options.Alpha;
    convergenceThreshold = Options.Tolerance; % Set a threshold for convergence

    % Partition the data into batches
    m = Options.BatchSize; % Batch size
    numBatches = ceil(numPoints / m);
    
    % Calculate the number of points after padding
    numPointsPadded = numBatches * m;
    
    % Initialize log-likelihood
    oldLogLikelihood = -inf;

    % Preallocate log_lh
    Log_Likelihood = gpuArray.zeros(numPoints, numClusters);

    % Compute constant term
    constTerm = dimFeatures*log(2*pi)/2;
    
    weights = gpuArray(ones(1, numClusters) / numClusters); % Equal weights
    warning("off")
    [~, mus] = kmeans(gpuArray(data), numClusters, 'MaxIter',numIterations); % Initialize means using kmeans
    Sigmas = gpuArray(ones(numClusters, dimFeatures)); % Unit variances
    
    for j = 1:numClusters


        log_prior = log(weights(j));

        logDetSigma = sum(log(Sigmas(j, :)));

        L = sqrt(Sigmas(j, :));

        Log_Likelihood(:,j) = sum(bsxfun(@rdivide, bsxfun(@minus, ...
            data, mus(j,:)), L).^2, 2);

        Log_Likelihood(:,j) = -0.5*(Log_Likelihood(:,j) + logDetSigma);

        Log_Likelihood(:,j) = Log_Likelihood(:,j) + log_prior - constTerm;

    end

    MaxLogLikelihood = max(Log_Likelihood,[],2);
    Responsibilities = exp(Log_Likelihood-MaxLogLikelihood);

    for i = 1 : numIterations
    
        % Create a random order in the data
        randomOrder = gpuArray([randperm(numPoints), 1:(numPointsPadded - numPoints)]);
    
        % Create batches of indices
        batches = reshape(randomOrder(1:numPointsPadded), m, numBatches);
    
        % Create a random order for the batches
        batchOrder = gpuArray(randperm(numBatches));
    
        for j = 1 : numBatches
    
            eta_k = (k+2)^(-a);
    
            batchIndices = unique(batches(:, batchOrder(j)));
            
            % Get the current batch of responsibilities
            currentBatch = Responsibilities(batchIndices,:);
    
            % Apply one-hot encoding 
            one_hot = one_hot_encoding(currentBatch);
    
            % Calculate the NEW responsibilities - Inference
            s_i = currentBatch .* one_hot;

            % Update and normalize responsibilities
            Responsibilities(batchIndices, :) = ((1-eta_k)*Responsibilities(batchIndices, :) ...
                                                + (eta_k * s_i)) ./ ...
                                                sum((1-eta_k)*Responsibilities(batchIndices, :) + ...
                                                (eta_k * s_i), 2);
            k = k + 1;
        end
        
        % Calculate log-likelihood after updating all batches
        Density = sum(Responsibilities, 2);
        Logpdf = log(Density) + MaxLogLikelihood;
        LogLikelihood = sum(Logpdf) / numPoints;
        

        % Check for convergence
        if abs(LogLikelihood - oldLogLikelihood) < convergenceThreshold || i == Options.MaxIterations

            fprintf('Converged in %d iterations\n', i);

                % Update best log likelihood
                bestLogLikelihood = LogLikelihood;

                %% Calculate negative log likelihood, AIC, and BIC
                NegLogLikelihood = -bestLogLikelihood;

                nParam = size(data, 2) + numClusters - 1 + numClusters * size(data, 2);
                AIC = 2 * nParam - 2 * bestLogLikelihood;
                BIC = nParam * log(numPoints) - 2 * bestLogLikelihood;

                % Update the struct with the new results
                GMM.NegLogLikelihood = NegLogLikelihood;
                GMM.AIC = AIC;
                GMM.BIC = BIC;
            
            break;
        end

        oldLogLikelihood = LogLikelihood;

    end

    function one_hot = one_hot_encoding(Responsibilities)
        % Get the number of classes
        numClasses = size(Responsibilities, 2);
    
        % For each data point, find the class with the highest responsibility
        [~, maxClass] = max(Responsibilities, [], 2);
        
        % Initialize the one-hot encoded matrix
        one_hot = gpuArray(zeros(size(Responsibilities, 1),numClasses));
        
        % Set the corresponding class to 1 using linear indexing
        one_hot(sub2ind(size(one_hot), (1:size(Responsibilities, 1))', maxClass)) = 1;
    end

end
