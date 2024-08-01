function GMM_Params = CalculateParamsNV(SIFT_data,RGB_data,Log_Likelihoods_SIFT,Log_Likelihoods_RGB)
    
    %% Create the parameters from the SIFT data
    for j = 1 : length(Log_Likelihoods_SIFT)

        Corr_LogLikelihood = gather(Log_Likelihoods_SIFT{j});

        get_numCluster = size(Corr_LogLikelihood,2);

        MaxCorrLogLikelihood = max(Corr_LogLikelihood,[],2);
        Responsibilities = gpuArray(exp(Corr_LogLikelihood - MaxCorrLogLikelihood));

        for k = 1 : get_numCluster

            Responsibilities_k = Responsibilities(:,k)';

            Nonzero_idx = Responsibilities_k > 0;

            SIFT_mus(k,:) = Responsibilities_k * SIFT_data / sum(Responsibilities_k);

            Centered_Data = SIFT_data(Nonzero_idx,:) - SIFT_mus(k,:);

            SIFT_Sigmas(k,:) = Responsibilities_k(Nonzero_idx) *...
                (Centered_Data.^2) / sum(Responsibilities_k(Nonzero_idx)) + 1e-6;

            SIFT_weights(k) = sum(Responsibilities_k) / size(SIFT_data,1);

        end

        GMM_Params(j).Training_SIFT_mus = gather(SIFT_mus);
        GMM_Params(j).Training_SIFT_Sigmas = gather(SIFT_Sigmas);
        GMM_Params(j).Training_SIFT_weights = gather(SIFT_weights);

    end

    %% Create the parameters from the RGB data
    for j = 1 : length(Log_Likelihoods_RGB)
            
        Corr_LogLikelihood = gather(Log_Likelihoods_RGB{j});
    
        get_numCluster = size(Corr_LogLikelihood,2);
    
        MaxCorrLogLikelihood = max(Corr_LogLikelihood,[],2);
        Responsibilities = gpuArray(exp(Corr_LogLikelihood - MaxCorrLogLikelihood));
    
        for k = 1 : get_numCluster
    
            Responsibilities_k = Responsibilities(:,k)';
        
            Nonzero_idx = Responsibilities_k > 0;
        
            RGB_mus(k,:) = Responsibilities_k * RGB_data / sum(Responsibilities_k);
        
            Centered_Data = RGB_data(Nonzero_idx,:) - RGB_mus(k,:);
        
            RGB_Sigmas(k,:) = Responsibilities_k(Nonzero_idx) *...
                (Centered_Data.^2) / sum(Responsibilities_k(Nonzero_idx)) + 1e-6;
        
            RGB_weights(k) = sum(Responsibilities_k) / size(RGB_data,1);
    
        end
    
        GMM_Params(j).Training_RGB_mus = gather(RGB_mus);
        GMM_Params(j).Training_RGB_Sigmas = gather(RGB_Sigmas);
        GMM_Params(j).Training_RGB_weights = gather(RGB_weights);

    end

    

end