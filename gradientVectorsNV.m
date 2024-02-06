function Fisher_Kernel = gradientVectorsNV(GMM_Params,Log_Likelihoods_SIFT,Log_Likelihoods_RGB)
    
    dimFeatures = size(GMM_Params(1).RGB_mus,2);
    numClusters = length(GMM_Params);
    Fisher_Kernel = [];
    
    
    %% Gradient Vectors for the SIFT features
    for Cluster = 1 : numClusters
        
        % Create the Fisher score
        gradient_vector = horzcat(GMM_Params(Cluster).SIFT_mus, ...
                                           GMM_Params(Cluster).SIFT_Sigmas);


        gradient_vector = reshape(gradient_vector,1,[]);
        % Create the Fisher kernel
        % Fish_fish = gradient_vector{Cluster}'*(eye(Cluster)\gradient_vector{Cluster});
        
        % Update the Fisher Information Matrix
        Fisher_Kernel = gradient_vector;


    end

        
end