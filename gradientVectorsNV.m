function Fisher_Information_Matrix = gradientVectorsNV(GMM_Params,Log_Likelihoods_SIFT,Log_Likelihoods_RGB)
    
    numClusters = length(GMM_Params);
    gradient_vector = cell(1,numClusters);
    Fisher_Information_Matrix = zeros(numClusters, numClusters);
    
    %% Gradient Vectors for the SIFT features
    for Cluster = 2 : numClusters

        Corr_LogLikelihood = gather(Log_Likelihoods_SIFT{Cluster});
    
        get_numCluster = size(Corr_LogLikelihood,2);
    
        MaxCorrLogLikelihood = max(Corr_LogLikelihood,[],2);
        Responsibilities = gpuArray(exp(Corr_LogLikelihood - MaxCorrLogLikelihood));

        gradient_vector{Cluster} = horzcat(GMM_Params(Cluster).SIFT_mus, ...
                                           GMM_Params(Cluster).SIFT_Sigmas);

        % Compute the gradient of the log-likelihood function
        gradient = Responsibilities * gradient_vector{Cluster};

        % Update the Fisher Information Matrix
        Fisher_Information_Matrix = Fisher_Information_Matrix + gradient' * gradient;

    end

    % Normalize the Fisher Information Matrix
    Fisher_Information_Matrix = Fisher_Information_Matrix / numClusters;
        
end