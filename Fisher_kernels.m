clear;clc;close all

%% Load the YOLOv4 Object Detector
yolo = yolov4ObjectDetector("csp-darknet53-coco");

%% Load the data 
filelocation = uigetdir;
imds = imageDatastore(filelocation);

%% Assign Labels

new_imds = assignLabels(imds,yolo);

%% Feature Extraction

% First step is to resize all the images
i=0;
reset(new_imds)
while hasdata(new_imds)
    i=i+1;
    im = read(new_imds);
    [rows(i),cols(i)] = size(im);
end

% Find the minimum dimensions of the images
maxrows = max(rows);
maxcols = max(cols);

imgSizeThresh = [maxrows,maxcols];

% Resize all the images based on the minimum dimensions using bilinear
% interpolation
resized_imds = transform(new_imds,@(x)imresize(x,imgSizeThresh,"bilinear"));

reset(resized_imds)
i=0;

%% Extract SIFT features from all the images
while hasdata(resized_imds)
    i=i+1;
    I = read(resized_imds);
    points = detectSIFTFeatures(im2gray(I));
    features(i).SIFTfeatures = extractFeatures(im2gray(I),points,"Method", ...
                                                              "SIFT");    
end

%% Extract low-level RGB statistic features from all the images

reset(resized_imds)
i=0;
while hasdata(resized_imds)
    i=i+1;
    I = im2double(read(resized_imds));
    
    % Divide the image into 2x8 grid
    [rows, cols, ~] = size(I);
    gridRows = 2;
    gridCols = 4;
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
