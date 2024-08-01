function Total_Fisher_Kernel = FisherEncodingNV(GMM_Params, Features)

    arguments

        GMM_Params {mustBeUnderlyingType(GMM_Params,'struct')}
        Features {mustBeUnderlyingType(Features,'struct')}

    end

    dimFeatures = size(Features(1).reduced_RGB_features, 2);
    numClusters = length(GMM_Params);
    numImages = length(Features);

    % Define chunk size (adjust this based on your memory constraints)
    chunkSize = 100;  % Process 100 images at a time

    % Initialize the result matrices
    SIFT_Fisher_Kernel = zeros(numImages, 2*dimFeatures*numClusters);
    RGB_Fisher_Kernel = zeros(numImages, 2*dimFeatures*numClusters);

    GMM_Params_Constant = parallel.pool.Constant(GMM_Params);

    %% Process SIFT features in chunks
    tic
    for chunkStart = 1:chunkSize:numImages
        chunkEnd = min(chunkStart + chunkSize - 1, numImages);
        currentChunk = chunkStart:chunkEnd;
        chunkSize = length(currentChunk);
        
        % Extract SIFT features for the current chunk
        chunkSIFTFeatures = arrayfun(@(x) Features(x).reduced_SIFT_features, currentChunk, 'UniformOutput', false);
        
        chunk_SIFT_Fisher_Kernel = zeros(chunkSize, 2*dimFeatures*numClusters);
        
        parfor i = 1:chunkSize
            numImage = currentChunk(i);
            fprintf("Now on SIFT Encoding in image: %d of %d \n", numImage, numImages);
            
            chunk_SIFT_Fisher_Kernel(i, :) = processFeatures(GMM_Params_Constant.Value, chunkSIFTFeatures{i}, numClusters, dimFeatures, 'SIFT');
        end
        
        SIFT_Fisher_Kernel(currentChunk, :) = chunk_SIFT_Fisher_Kernel;
    end
    SIFT_encoding_time = toc;
    fprintf("Finished encoding SIFT features. Time: %f \n", SIFT_encoding_time);

    %% Process RGB features in chunks
    tic
    for chunkStart = 1:chunkSize:numImages
        
        chunkEnd = min(chunkStart + chunkSize - 1, numImages);
        currentChunk = chunkStart:chunkEnd;
        chunkSize = length(currentChunk);
        
        % Extract RGB features for the current chunk
        chunkRGBFeatures = arrayfun(@(x) Features(x).reduced_RGB_features, currentChunk, 'UniformOutput', false);
        
        chunk_RGB_Fisher_Kernel = zeros(chunkSize, 2*dimFeatures*numClusters);
        
        parfor i = 1:chunkSize

            numImage = currentChunk(i);
            fprintf("Now on RGB Encoding on image: %d of %d \n", numImage, numImages);
            
            chunk_RGB_Fisher_Kernel(i, :) = processFeatures(GMM_Params_Constant.Value, ...
                                              chunkRGBFeatures{i}, numClusters, dimFeatures, 'RGB');
        
        end
        
        RGB_Fisher_Kernel(currentChunk, :) = chunk_RGB_Fisher_Kernel;

    end

    RGB_encoding_time = toc;
    fprintf("Finished encoding RGB features. Time: %f \n", RGB_encoding_time);

    Total_Fisher_Kernel = [SIFT_Fisher_Kernel; RGB_Fisher_Kernel];

end

function gradient_vector = processFeatures(GMM_Params, currentFeatureMatrix, numClusters, ...
                                                                           dimFeatures, featureType)

    gradient_vector = zeros(1, 2*dimFeatures*numClusters);
    currentFeatureMatrix = gpuArray(currentFeatureMatrix);

    for Cluster = 1:numClusters

        % Get the current cluster's parameters
        Means = gpuArray(GMM_Params(Cluster).(['Training_' featureType '_mus']));
        Sigmas = gpuArray(GMM_Params(Cluster).(['Training_' featureType '_Sigmas']));
        Weights = gpuArray(GMM_Params(Cluster).(['Training_' featureType '_weights']));

        % Initialize the covariance matrices
        covMatrices = gpuArray(zeros(1, dimFeatures, Cluster));

        for i = 1:Cluster

            covMatrices(1, :, i) = Sigmas(i, :);

        end

        % Create a gmdistribution object
        gmModel = gmdistribution(Means, covMatrices);

        ImagePosterior = posterior(gmModel, currentFeatureMatrix);

        % Compute F_k for the means
        
        F_k_means = (currentFeatureMatrix - Means(Cluster, :)) ./ (Sigmas(Cluster, :).^2);
        F_k_means = sum(F_k_means .* ImagePosterior(:, Cluster) * Weights(Cluster), 1);

        f_mui = (size(currentFeatureMatrix, 1) * Weights(Cluster)) ./ Sigmas(Cluster, :).^2;
        
        F_k_means_normalized = F_k_means ./ f_mui;

        % Compute F_k for the variances
        F_k_variances = ((currentFeatureMatrix - Means(Cluster, :)).^2) ./ (Sigmas(Cluster, :).^3) ...
        - 1 ./ (Sigmas(Cluster, :));
        F_k_variances = sum(F_k_variances .* ImagePosterior(:, Cluster) * Weights(Cluster), 1);

        f_si = 2 * size(currentFeatureMatrix, 1) * Weights(Cluster) ./ Sigmas(Cluster, :).^2;

        F_k_variances_normalized = F_k_variances ./ f_si;

        % Assign F_k_means and F_k_variances to gradient_vector
        gradient_vector(1, (Cluster-1)*2*dimFeatures+1 : Cluster*2*dimFeatures) = ...
        [F_k_means_normalized, F_k_variances_normalized];

    end

end