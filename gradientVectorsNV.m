function Fisher_Kernel = gradientVectorsNV(GMM_Params,features)
    
    dimFeatures = size(features(1).reduced_RGB_features,2);
    numClusters = length(GMM_Params);
    Fisher_Kernel = zeros(length(features),2*dimFeatures*numClusters);

    %% Gradient Vectors for the SIFT features

    for numImages = 1 : length(features)

        % Initialize the gradient vector
        gradient_vector = zeros(1, 2*dimFeatures*128);


        for Cluster = 1 : numClusters

            % Get the current clusters parameters
            Means = GMM_Params(Cluster).SIFT_mus;
            Sigmas = GMM_Params(Cluster).SIFT_Sigmas;
            Weights = GMM_Params(Cluster).SIFT_weights;

            % Get the current feature matrix
            currentFeatureMatrix = features(numImages).reduced_SIFT_features;
            
            % Initialize the covariance matrices
            covMatrices = zeros(1, dimFeatures, numClusters);

            % Loop over each cluster

            for i = 1:Cluster
                % Create a square diagonal matrix from the covariance vector
                covMatrices(1, :, i) = GMM_Params(Cluster).SIFT_Sigmas(i, :);
            end
            
            % Create a gmdistribution object
            gmModel = gmdistribution(Means, covMatrices);

            ImagePosterior = posterior(gmModel,currentFeatureMatrix);
            
            F_k = (currentFeatureMatrix - Means(Cluster, :)) ./ Sigmas(:, :, Cluster);

            % Weight by posterior probability and cluster weight
            F_k = F_k .* posterior(:, k) / sqrt(Weights(k));

            % Sum over all descriptors
            F_k = sum(F_k, 1);

            % Concatenate to gradient vector
            gradient_vector(1, (Cluster-1)*2*dimFeatures+1:Cluster*2*dimFeatures) = F_k;


        end

        % Update the Fisher Information Matrix
            Fisher_Kernel = gradient_vector;

    end

    
        
end

% % Create the Fisher score
            % gradient_vector = horzcat(GMM_Params(Cluster).SIFT_mus, ...
            %     GMM_Params(Cluster).SIFT_Sigmas);
            %
            % gradient_vector = reshape(gradient_vector,1,[]);
            % Create the Fisher kernel
            % Fish_fish = gradient_vector{Cluster}'*(eye(Cluster)\gradient_vector{Cluster});
