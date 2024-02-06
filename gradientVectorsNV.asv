function gradientVectors = gradientVectorsNV(GMM_Params,Log_Likelihoods_SIFT,Log_Likelihoods_RGB)
    
    Fisher_Information_Matrix = [];
    numClusters = length(GMM_Params);
    gradient_vector = cell(1,numClusters);
    
    %% Gradient Vectors for the SIFT features
    for Cluster = 2 : numClusters

        Corr_LogLikelihood = gather(Log_Likelihoods_SIFT{Cluster});
    
        get_numCluster = size(Corr_LogLikelihood,2);
    
        MaxCorrLogLikelihood = max(Corr_LogLikelihood,[],2);
        Responsibilities = gpuArray(exp(Corr_LogLikelihood - MaxCorrLogLikelihood));

        gradient_vector{Cluster} = horzcat(GMM_Params(Cluster).SIFT_mus, ...
                                           GMM_Params(Cluster).SIFT_Sigmas);

        Expected_value = sum(Responsibilities * gradient_vector{Cluster} * ...
                                                gradient_vector{Cluster}');
       
        Fisher_Information_Matrix = [Fisher_Information_Matrix Expected_value];

    end

        
end