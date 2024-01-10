function features = extractImageFeatures(datastore,optional)

%% Description 
% -------------------------------------------------------------------------
% A function that extracts image features from an image datastore. The 
% function first resizes all images to the same size. Then, it extracts 
% SIFT features and low-level RGB statistic features from each image. The 
% images are divided into a grid, and features are extracted from each grid
% cell. Finally, the function applies PCA to reduce the number of features.

% Inputs: 
% -------------------------------------------------------------------------
% => datastore: The imageDatastore that contains the raw data. 
% 
% => optional: A structure that contains optional parameters for the 
%              function. It includes the number of grid rows and columns 
%              for dividing the image, and the number of features to retain
%              after PCA.

% Outputs: 
% ------------------------------------------------------------------------- 
% => features: A structure array that contains the extracted features for 
%              each image. Each element of the array corresponds to one 
%              image and contains the SIFT features, RGB statistic 
%              features, and the reduced versions of these features after 
%              PCA.
% 
% Details: 
% -------------------------------------------------------------------------
% The function works in several steps. First, it resizes all images to the 
% same size. Then, it divides each image into a grid and extracts SIFT 
% features from each grid cell. If no SIFT points are found in a cell, the
% function continues with the next cell. The SIFT features from all cells 
% are concatenated vertically.
%
% In the next step, the function calculates low-level RGB statistic 
% features for each grid cell. These include the mean, standard deviation,
% skewness, and kurtosis of the RGB values. The statistics for all cells 
% are reshaped into a 1-by-96 vector.
%
% Finally, the function applies PCA to the SIFT features and RGB statistic 
% features separately, and retains a specified number of principal 
% components. The reduced features are added to the output structure.


arguments (Input)

    datastore          {mustBeUnderlyingType(datastore, ...
                                                ['matlab.io.datastore.' ...
                                                'ImageDatastore'])}



    optional.gridRows  {mustBePositive,mustBeNonempty,...
                        mustBeReal,mustBeInteger} = 2

    optional.gridCols  {mustBePositive,mustBeNonempty,...
                        mustBeReal,mustBeInteger} = 4

    optional.Feature_reduction {mustBePositive,mustBeNonempty,...
                                mustBeReal,mustBeInteger} = 50
end

% First step is to resize all the images
i=0;
reset(datastore)
rows = zeros(length(datastore.Files),1);
cols = zeros(length(datastore.Files),1);
while hasdata(datastore)
    i=i+1;
    im = read(datastore);
    [rows(i),cols(i)] = size(im);
end

% Find the minimum dimensions of the images
maxrows = max(rows);
maxcols = max(cols);

imgSizeThresh = [maxrows,maxcols];

% Resize all the images based on the minimum dimensions using bilinear
% interpolation
resized_imds = transform(datastore,@(x)imresize(x,imgSizeThresh,"bilinear"));

reset(resized_imds)
i=0;

%% Extract SIFT features from all the images
while hasdata(resized_imds)
    i=i+1;
    I = read(resized_imds);

    % Divide the image into 2-by-4 grid
    [rows, cols, ~] = size(I);
    gridRows = optional.gridRows;
    gridCols = optional.gridCols;

    regionHeight = floor(rows/gridRows);
    regionWidth = floor(cols/gridCols);

    % Calculate statistics for each region
    region_SIFT_features = [];
    for r = 1:gridRows
        for c = 1:gridCols
            % Extract the region
            region = I((r-1)*regionHeight+1:r*regionHeight, ...
                (c-1)*regionWidth+1:c*regionWidth, :);

            % Find the SIFT points of the region
            points = detectSIFTFeatures(im2gray(region));

            if points.Count == 0

                continue

            else
                
                SIFT_features = extractFeatures(im2gray(region), ...
                    points,"Method","SIFT");

                region_SIFT_features = vertcat(region_SIFT_features,...
                                               SIFT_features);
                
            end
        end
    end
    features(i).SIFT_Features = region_SIFT_features;
end


%% Extract low-level RGB statistic features from all the images

reset(resized_imds)
i=0;

while hasdata(resized_imds)
    i=i+1;
    I = im2double(read(resized_imds));

    % Divide the image into 2-by-4 grid
    [rows, cols, ~] = size(I);
    gridRows = optional.gridRows;
    gridCols = optional.gridCols;

    regionHeight = floor(rows/gridRows);
    regionWidth = floor(cols/gridCols);
    regionStats_reshaped = zeros(8,12);

    % Calculate statistics for each region
    regionIndex = 1;
    for r = 1:gridRows
        for c = 1:gridCols
            % Extract the region
            region = I((r-1)*regionHeight+1:r*regionHeight, ...
                (c-1)*regionWidth+1:c*regionWidth, :);

            % Calculate statistics for the region
            regionStats = [mean(region, [1,2]), ...
                std(region, 1, [1,2]), ...
                skewness(region, 1, [1,2]), ...
                kurtosis(region, 1, [1,2])];

            regionStats= reshape(regionStats,[1 12]);

            regionStats_reshaped(regionIndex,:) = regionStats;
            % Increment the region index
            regionIndex = regionIndex + 1;
        end
    end

    features(i).RGBfeatures = reshape(regionStats_reshaped,[1,96]);
end

%% Apply PCA to reduce to the optional number of features

for i = 1: length(features)

    [~,SIFT_score] = pca(features(i).SIFT_Features);

    % Select the first 50 principal components
    reduced_SIFT_data = SIFT_score(:, 1:optional.Feature_reduction);
    features(i).reduced_SIFT_features = reduced_SIFT_data;

    [~,RGB_score] = pca(features(i).RGBfeatures);

    % Select the first 50 principal components
    reduced_RGB_data = RGB_score(:, 1:optional.Feature_reduction);
    features(i).reduced_RGB_features = reduced_RGB_data;

end
