clear;clc;close all

%%
yolo = yolov4ObjectDetector("csp-darknet53-coco");
%%
filelocation = uigetdir;
imds = imageDatastore(filelocation);
%%
i=0;reset(imds)
scores= zeros(length(imds.Files),1);
labelsYolo = zeros(length(imds.Files),1);

totalFiles = length(imds.Files);
loadingIcons = ['-', '\', '|', '/']; % Loading icon sequence

tic
% Create a waitbar
h = waitbar(0, 'Processing...'); warning("off")

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

% Find the Labels with Review 
ind = find(Labels == "Review");

% Find the images that correspond to those indexes
Review_images = imds.Files(ind);