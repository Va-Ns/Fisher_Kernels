clear;clc;close all

filelocation = uigetdir;
imds = imageDatastore(filelocation);

yolo = yolov4ObjectDetector("csp-darknet53-coco");

i=0;reset(imds)
scores= zeros(length(imds.Files),1);
labelsYolo = zeros(length(imds.Files),1);
tic
while hasdata(imds)
    i=i+1;
    img = read(imds);
    [~,scoresYolo,labelsYolo] = detect(yolo,img,"Threshold",0.90,'ExecutionEnvironment','gpu');
    
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
    
    iter = mod(i,100);
    fprintf('Now in iteration: %d',iter)

end
Labeling_time = toc
