clear;clc;close all

%% Load the YOLOv4 Object Detector
yolo = yolov4ObjectDetector("csp-darknet53-coco");

%% 
filelocation = uigetdir;
imds = imageDatastore(filelocation);

%% Perform Object Detection

i=0;
reset(imds)

scores= zeros(length(imds.Files),1);
labelsYolo = zeros(length(imds.Files),1);

totalFiles = length(imds.Files);
loadingIcons = ['-', '\', '|', '/']; % Loading icon sequence

% Create a waitbar
h = waitbar(0, 'Processing...'); warning("off")

tic

while hasdata(imds)
    i=i+1;
    img = read(imds);
    [~,scoresYolo,labelsYolo] = detect(yolo,img,"Threshold",0.90, ...
                                       'ExecutionEnvironment','gpu');
    
    if isempty(scoresYolo)

        scores(i) = 0;
        Labels(i) = categorical("Review");

    elseif numel(labelsYolo) > 1

        [scores(i),ind] = max(scoresYolo,[],"all");
        Labels(i) = unique(labelsYolo(ind));

    else

        scores(i) = scoresYolo;
        Labels(i) = labelsYolo;

    end
    
    % Calculate percentage of completion
    percentComplete = i / totalFiles;
    
    % Update waitbar
    waitbar(percentComplete, h, sprintf('Processing %s \n %.2f%% complete', ...
     loadingIcons(mod(i-1, numel(loadingIcons)) + 1), percentComplete*100));

    
end

% Close waitbar
close(h)
Labeling_time = toc

%% Clean the data

% Put the Labels to the corresponding place in the structure-type variable
imds.Labels = Labels;

% Find the Labels with Review label
ind = find(Labels == "Review");

% Find the images that correspond to those indexes and create a new image 
% datastore that doesn't include them. Also remove the labels of those 
% indices from the labels array

delimg = false(size(imds.Files)); % Initialize delimg array

for i = 1:length(ind)
    delimg(matches(imds.Files, imds.Files{ind(i),1})) = true;
end
new_imds = subset(imds, ~delimg);

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
