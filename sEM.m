function GMM = sEM(data,numClusters,Options)
    
    arguments 

        data (:,:) double {mustBeReal, mustBeFinite}
        numClusters (1,1) double {mustBeInteger, mustBePositive,...
                                  mustBeNonempty,mustBeNonzero,mustBeNonmissing}
        Options.MaxIterations (1,1) double {mustBeInteger, mustBePositive} = 100;
        Options.Tolerance (1,1) double {mustBeReal, mustBeFinite} = 1e-6;
        Options.Alpha     (1,1) double  {mustBeInRange(Options.Alpha,0.5,1),...
                                        mustBePositive, mustBeFloat} = 0.8
        Options.BatchSize     (1,1) double {mustBeInRange(Options.BatchSize, ...
                                            1,1000000),mustBePositive,... 
                                            mustBeInteger} = 10000
                                        
    end
    
    [numPoints, dimFeatures] = size(data);
       
    k = 0;
    numIterations = Options.MaxIterations;
    a = Options.Alpha;
    convergenceThreshold = Options.Tolerance; % Set a threshold for convergence

    %% Partition the data into batches
    m = Options.BatchSize; 
    numBatches = ceil(numPoints / m);
    
    %% Calculate the number of points after padding
    numPointsPadded = numBatches * m;
    
    %% Initialize log-likelihood
    oldLogLikelihood = -inf;

    %% Preallocate log_lh
    Log_Likelihood = gpuArray.zeros(numPoints, numClusters);

    %% Compute constant term
    constTerm = dimFeatures*log(2*pi)/2;
    
    %% Perform the initialization of the Responsibilities
    weights = gpuArray(ones(1, numClusters) / numClusters); % Equal weights
    warning("off")
    [~, mus] = kmeans(gpuArray(data), numClusters, 'MaxIter',numIterations); % Initialize means using kmeans
    Sigmas = gpuArray(ones(numClusters, dimFeatures)); % Unit variances
    
    for j = 1:numClusters

        log_prior = log(weights(j));

        logDetSigma = sum(log(Sigmas(j, :)));

        L = sqrt(Sigmas(j, :));

        Log_Likelihood(:,j) = sum(bsxfun(@rdivide, bsxfun(@minus, ...
            data, gather(mus(j,:))), L).^2, 2);

        Log_Likelihood(:,j) = -0.5*(Log_Likelihood(:,j) + logDetSigma);

        Log_Likelihood(:,j) = Log_Likelihood(:,j) + log_prior - constTerm;

    end

    MaxLogLikelihood = max(Log_Likelihood,[],2);
    Responsibilities = exp(Log_Likelihood-MaxLogLikelihood);
    % Vectorized calculation of log-likelihoods
    % log_prior = log(weights);
    % logDetSigma = sum(log(Sigmas), 2);
    % L = sqrt(Sigmas);
    % Log_Likelihood = zeros(size(data, 1), size(mus, 1));  % Initialize Log_Likelihood
    % 
    % for k = 1:size(mus, 1)
    %     Log_Likelihood(:, k) = -0.5 * sum(((data - mus(k, :)) ./ ...
    %                            L(k, :)).^2, 2) + log_prior(k) - constTerm;
    % end
    % 
    % % Vectorized calculation of responsibilities
    % MaxLogLikelihood = max(Log_Likelihood,[],2);
    % Responsibilities = exp(bsxfun(@minus, Log_Likelihood, MaxLogLikelihood));

%% Apply the stepwise Expectation Maximization

    for i = 1 : numIterations
    
        % Create a random order in the data
        randomOrder = gpuArray([randperm(numPoints), 1:(numPointsPadded - numPoints)]);
    
        % Create batches of indices
        batches = reshape(randomOrder(1:numPointsPadded), m, numBatches);
    
        % Create a random order for the batches
        batchOrder = gpuArray(randperm(numBatches));
    
        for j = 1 : numBatches % Here the examples are the batches of data
    
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
                                                + (eta_k * s_i));
            k = k + 1;
        end

        % M-step updates
        sumResponsibilities = sum(Responsibilities, 1);
        mus = (Responsibilities' * data) ./ sumResponsibilities';
        for j = 1:numClusters
            diff = data - mus(j, :);
            Sigmas(j, :) = (Responsibilities(:, j)' * (diff .^ 2)) / sumResponsibilities(j) + 1e-6;
        end
        weights = sumResponsibilities / numPoints;
        
        % Calculate log-likelihood after updating all batches
        Density = sum(Responsibilities, 2);
        Logpdf = log(Density) + MaxLogLikelihood;
        LogLikelihood = sum(Logpdf) / numBatches;
        
        % Calculate the sum of responsibilities for each cluster
        % sumResponsibilities = sum(Responsibilities, 1);
        %
        % Calculate the new means for each cluster
        % mus = (Responsibilities' * data) ./ sumResponsibilities';
        %
        % Initialize an empty matrix to store the centered data
        % Centered_Data = zeros(size(data));
        %
        % Loop over each cluster
        % for j = 1:size(mus, 1)
            % Subtract the mean of the j-th cluster from the data
            % Centered_Data(j, :) = data(j, :) - mus(j, :);
        % end
        % Calculate the new variances for each cluster
        % Sigmas = (Responsibilities' * (Centered_Data.^2)) ./ sumResponsibilities' + 1e-6;
        %
        % Calculate the new weights for each cluster
        % weights = sumResponsibilities / numPoints;
        if abs(LogLikelihood - oldLogLikelihood) < convergenceThreshold || i == Options.MaxIterations

                fprintf("Converged in iteration %d\n",i)

                % Update best log likelihood
                bestLogLikelihood = LogLikelihood;
                
                %% Place the best found statistics in the struct
                NegLogLikelihood = -bestLogLikelihood;

                % Update the struct with the new results
                GMM.NegLogLikelihood = NegLogLikelihood;
                GMM.Log_Likelihood = tall(gather(Log_Likelihood));
              
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
