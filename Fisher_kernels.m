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

%%
% Put the Labels to the corresponding structular place
imds.Labels = Labels;

% Find the Labels with Review 
ind = find(Labels == "Review");

% Find the images that correspond to those indexes and create a new image 
% datastore that doesn't include them and also remove the labels from the 
% labels array


delimg = false(size(imds.Files)); % Initialize delimg array

for i = 1:length(ind)
    delimg(matches(imds.Files, imds.Files{ind(i),1})) = true;
end
new_imds = subset(imds, ~delimg);

