tic
for i = 1 : 5


    fprintf("Number of cluster:%d \n",i)
    GMMs{i} = sEM(FeatureMatrix.Reduced_SIFT_Features_Matrix, i); 
    logLikelihoods(i) = GMMs{i}.NegLogLikelihood;
    AICs(i) = GMMs{i}.AIC;
    BICs(i) = GMMs{i}.BIC;

    fprintf(" >> Negative Log-Likelihood:%e\n ",logLikelihoods(i))   
    % hold on
    % plot(i,logLikelihoods(i),"o");
    % xlabel('Iterations'); ylabel('Log-Likelihood');
    % titleStr = sprintf('Log-Likelihood vs Iterations');
    % title(titleStr);
    % drawnow
    
end
five_clusters_time = toc
