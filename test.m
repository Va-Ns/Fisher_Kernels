tic
for i = 1 : numModels

    fprintf("Number of cluster:%d \n",i)

    GMMs = sEM(descriptors, i,"Alpha",0.5,"BatchSize",100,"MaxIterations",10); 
    logLikelihoods(i) = GMMs.NegLogLikelihood;
    Log_Likelihoods{i} = GMMs.Log_Likelihood;

    fprintf(" >> Negative Log-Likelihood:%e\n ",logLikelihoods(i))   
    
end
sEM_RGB_time = toc

%% Calculate params 
for j = 1 : length(Log_Likelihoods)

    Corr_LogLikelihood = gather(Log_Likelihoods{j});

    get_numCluster = size(Corr_LogLikelihood,2);

    MaxCorrLogLikelihood = max(Corr_LogLikelihood,[],2);
    Responsibilities = gpuArray(exp(Corr_LogLikelihood - MaxCorrLogLikelihood));

    parfor k = 1 : get_numCluster

        Responsibilities_k = Responsibilities(:,k)';

        Nonzero_idx = Responsibilities_k > 0;

        Mus(k,:) = Responsibilities_k * descriptors / sum(Responsibilities_k);

        Centered_Data = descriptors(Nonzero_idx,:) - Mus(k,:);

        Sigmas(k,:) = Responsibilities_k(Nonzero_idx) *...
            (Centered_Data.^2) / sum(Responsibilities_k(Nonzero_idx)) + 1e-6;

        Weights(k) = sum(Responsibilities_k) / size(descriptors,1);

    end

    Train_GMM_Params(j).Training_mus = gather(Mus);
    Train_GMM_Params(j).Training_Sigmas = gather(Sigmas);
    Train_GMM_Params(j).Training_weights = gather(Weights);

end

%%
Training_Fisher_Kernel = [];

for i = 1 : length(Indices.Train_Indices)

    fprintf("Now on Training Encoding in image: %d of %d \n", i, length(Indices.Train_Indices))
    
    % Initialize the gradient vector
    gradient_vector = [];

    for Cluster = 1 : numModels

        % Get the current clusters parameters
            Means = gpuArray(Train_GMM_Params(Cluster).Training_mus);
            Sigmas = gpuArray(Train_GMM_Params(Cluster).Training_Sigmas);
            Weights = gpuArray(Train_GMM_Params(Cluster).Training_weights);

            % Get the current feature matrix
            currentFeatureMatrix = gpuArray(features{Indices.Train_Indices(i)});
            
            for k = 1:Cluster
    
                % Create a square diagonal matrix from the covariance vector
                covMatrices(1, :, k) = Sigmas(k, :);
    
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
            gradient_vector = [gradient_vector;[F_k_means_normalized,F_k_variances_normalized]];
            
    end
    gradient_vector = reshape(gradient_vector,1,2*128*128);
    Training_Fisher_Kernel = [Training_Fisher_Kernel;gradient_vector];

end
%%
Testing_Fisher_Kernel = [];

for i = 1 : length(Indices.Test_Indices)

    fprintf("Now on Testing Encoding in image: %d of %d \n", i, length(Indices.Test_Indices))
    
    % Initialize the gradient vector
    gradient_vector = [];

    for Cluster = 1 : numModels

        % Get the current clusters parameters
            Means = gpuArray(Train_GMM_Params(Cluster).Training_mus);
            Sigmas = gpuArray(Train_GMM_Params(Cluster).Training_Sigmas);
            Weights = gpuArray(Train_GMM_Params(Cluster).Training_weights);

            % Get the current feature matrix
            currentFeatureMatrix = gpuArray(features{Indices.Test_Indices(i)});
            
            for l = 1:Cluster
    
                % Create a square diagonal matrix from the covariance vector
                covMatrices(1, :, l) = Sigmas(l, :);
    
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
            gradient_vector = [gradient_vector;[F_k_means_normalized,F_k_variances_normalized]];
            
    end
    gradient_vector = reshape(gradient_vector,1,2*128*128);
    Testing_Fisher_Kernel = [Testing_Fisher_Kernel;gradient_vector];

end
%
t = templateSVM('SaveSupportVectors',true,'Type','classification');
[Model1,HyperparameterOptimizationResults] = fitcecoc(Training_Fisher_Kernel, ...
    Trainds.Labels,"Learners",t,"Coding", "onevsall", ...
    'OptimizeHyperparameters','all', 'HyperparameterOptimizationOptions', ...
    struct('Holdout',0.1,'MaxObjectiveEvaluations',100));


[predictedLabels, scores]= predict(Model1,Testing_Fisher_Kernel);

confusionMatrix = confusionmat(Testds.Labels,predictedLabels);

accuracy = sum(diag(confusionMatrix)) / sum(confusionMatrix(:))
