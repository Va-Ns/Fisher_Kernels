function Total_Fisher_Kernel = FisherEncodingNV(GMM_Params, Features)

    arguments

        GMM_Params {mustBeUnderlyingType(GMM_Params,'struct')}

        Features {mustBeUnderlyingType(Features,'struct')}

    end

    dimFeatures = size(Features(1).reduced_RGB_features,2);
    numClusters = length(GMM_Params);

    %% Initialization
    SIFT_Fisher_Kernel = zeros(length(Features),2*dimFeatures*numClusters);
    RGB_Fisher_Kernel  = zeros(length(Features),2*dimFeatures*numClusters);
    
    %% Gradient Vectors for the SIFT features
   
    tic
    for numImages = 1 : length(Features)

        fprintf("Now on SIFT Encoding in image: %d of %d \n", numImages, length(Features))
    
        % Initialize the gradient vector
        gradient_vector = zeros(1, 2*dimFeatures*numClusters);
    
        for Cluster = 1 : numClusters
    
            % Get the current clusters parameters
            Means = gpuArray(GMM_Params(Cluster).Training_SIFT_mus);
            Sigmas = gpuArray(GMM_Params(Cluster).Training_SIFT_Sigmas);
            Weights = gpuArray(GMM_Params(Cluster).Training_SIFT_weights);
    
            % Get the current feature matrix
            currentFeatureMatrix = gpuArray(Features(numImages).reduced_SIFT_features);
    
            % Initialize the covariance matrices
            covMatrices = gpuArray(zeros(1, dimFeatures, Cluster));
    
            % Loop over each cluster
    
            for i = 1:Cluster
    
                % Create a square diagonal matrix from the covariance vector
                covMatrices(1, :, i) = Sigmas(i, :);
    
            end
    
            % Create a gmdistribution object
            gmModel = gmdistribution(Means, covMatrices);
    
            ImagePosterior = posterior(gmModel,currentFeatureMatrix);
    
            % Compute F_k for the means
            F_k_means = (currentFeatureMatrix - Means(Cluster, :)) ...
                ./ (Sigmas(Cluster, :).^2);

            F_k_means = sum(F_k_means .* ImagePosterior(:, Cluster) * ...
                                                          Weights(Cluster),1);

            f_mui = (size(currentFeatureMatrix,1) * Weights(Cluster)) ./ ...
                                                    Sigmas(Cluster, :).^2;
            
            % Compute the normalized gradient of the mean parameter
            F_k_means_normalized = F_k_means ./ f_mui;

            % Compute F_k for the variances
            F_k_variances = ((currentFeatureMatrix - Means(Cluster, :)).^2)...
                ./ (Sigmas(Cluster, :).^3)...
              - 1 ./ (Sigmas(Cluster, :));

            F_k_variances = sum(F_k_variances .* ImagePosterior(:, Cluster)...
                                                        * Weights(Cluster),1);
                                                        
            f_si = 2 * size(currentFeatureMatrix,1) * Weights(Cluster) ./...
                                                            Sigmas(Cluster, :).^2;
                   
            % Compute the normalized gradient of the mean parameter
            F_k_variances_normalized = F_k_variances ./ f_si;
           
            % Assign F_k_means and F_k_variances to gradient_vector
            gradient_vector(1, (Cluster-1)*2*dimFeatures+1: ...
                              Cluster*2*dimFeatures) = [F_k_means_normalized,...
                                                        F_k_variances_normalized];

        end

         % Update the Fisher Information Matrix
         SIFT_Fisher_Kernel(numImages,:) = gradient_vector;

    end

    SIFT_encoding_time = toc;

    fprintf("Finished encoding SIFT features. Time: %f \n",SIFT_encoding_time);
    
    %% Gradient Vectors for the RGB features

    tic

    for numImages = 1 : length(Features)

        fprintf("Now on RGB Encoding on image: %d of %d \n", numImages, length(Features))

        % Initialize the gradient vector
        gradient_vector = zeros(1, 2*dimFeatures*numClusters);
    
        for Cluster = 1 : numClusters
    
            % Get the current clusters parameters
            Means = gpuArray(GMM_Params(Cluster).Training_RGB_mus);
            Sigmas = gpuArray(GMM_Params(Cluster).Training_RGB_Sigmas);
            Weights = gpuArray(GMM_Params(Cluster).Training_RGB_weights);
    
            % Get the current feature matrix
            currentFeatureMatrix = gpuArray(Features(numImages).reduced_RGB_features);
    
            % Initialize the covariance matrices
            covMatrices = gpuArray(zeros(1, dimFeatures, Cluster));
    
            % Loop over each cluster
    
            for i = 1:Cluster
    
                % Create a square diagonal matrix from the covariance vector
                covMatrices(1, :, i) = Sigmas(i, :);
    
            end
    
            % Create a gmdistribution object
            gmModel = gmdistribution(Means, covMatrices);

            ImagePosterior = posterior(gmModel,currentFeatureMatrix);
    
            % Compute F_k for the means
            F_k_means = (currentFeatureMatrix - Means(Cluster, :)) ...
                ./ (Sigmas(Cluster, :).^2);

            F_k_means = sum(F_k_means .* ImagePosterior(:, Cluster) * ...
                                                          Weights(Cluster),1);

            f_mui = (size(currentFeatureMatrix,1) * Weights(Cluster)) ./ ...
                                                    Sigmas(Cluster, :).^2;
            
            % Compute the normalized gradient of the mean parameter
            F_k_means_normalized = F_k_means ./ f_mui;

            % Compute F_k for the variances
            F_k_variances = ((currentFeatureMatrix - Means(Cluster, :)).^2)...
                                                 ./ (Sigmas(Cluster, :).^3)...
                                               - 1 ./ (Sigmas(Cluster, :));

            F_k_variances = sum(F_k_variances .* ImagePosterior(:, Cluster)...
                                                        * Weights(Cluster),1);

            f_si = 2 * size(currentFeatureMatrix,1) * Weights(Cluster) ./...
                                                     Sigmas(Cluster, :).^2;

            % Compute the normalized gradient of the mean parameter
            F_k_variances_normalized = F_k_variances ./ f_si;
    
            % Assign F_k_means and F_k_variances to gradient_vector
            gradient_vector(1, (Cluster-1)*2*dimFeatures+1: ...
                        Cluster*2*dimFeatures) = [F_k_means_normalized,...
                                                  F_k_variances_normalized];
                                                
                                                
        end      
        
        % Update the Fisher Information Matrix
        RGB_Fisher_Kernel(numImages,:) = gradient_vector;

    end

    RGB_encoding_time = toc;
    fprintf("Finished encoding RGB features. Time: %f \n",RGB_encoding_time);
    
    Total_Fisher_Kernel = [SIFT_Fisher_Kernel;RGB_Fisher_Kernel];

end