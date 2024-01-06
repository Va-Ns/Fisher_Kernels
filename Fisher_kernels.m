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
    
    remaining_iter = mod(i,100);
    if mod(i, 100) == 0
        fprintf('Remaining iterations: %d\n',length(imds.Files) - i);
    end

end
Labeling_time = toc
