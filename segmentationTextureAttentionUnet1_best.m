%% segmentationTextureAttentionUnet1_best.m
% Best Attention U-Net config (15 layers, SGDM, 50 epochs) on Case 1
% Measures training/inference time and computes additional metrics.

clc;

%% Paths (Windows)
cd('C:\SRP\TextureSegmentation\Texture-Segmentation-master\CODE');
dataSetDir  = 'C:\SRP\TextureSegmentation\Texture-Segmentation-master\CODE\';
dataSaveDir = 'C:\SRP\TextureSegmentation\Texture-Segmentation-master\CODE\Results\';

if ~exist(dataSaveDir,'dir'), mkdir(dataSaveDir); end

%% Load randenData and apply texture changes
load randenData
clear resRanden stdsRanden meansRanden fname ind edge error*

changetrainRanden1
changeDataRanden1

results_best_attn = [];

%% Only Case 1
currentCase = 1;

[rows,cols] = size(dataRanden{currentCase});
numClasses  = size(trainRanden{currentCase},3);

imageDir = fullfile(dataSetDir, 'trainingImages', sprintf('Case_%d',currentCase));
labelDir = fullfile(dataSetDir, 'trainingLabels', sprintf('Case_%d',currentCase));

imds  = imageDatastore(imageDir);

% Class names "T1","T2",...
clear classNames
for k = 1:numClasses
    classNames(k) = strcat("T", num2str(k));
end

labelIDs = 1:numClasses;
pxds = pixelLabelDatastore(labelDir, classNames, labelIDs);

%% ---- BEST CONFIG ----
numEpochs    = 50;
typeEncoder  = 'sgdm';
nameEncoder  = '1';
numLayersNet = 15;
nameLayers   = '15';

% Attention U-Net architecture (15-layer)
lgraph = attentionUnet15layers(numClasses);

% Training options
opts = trainingOptions(typeEncoder, ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',numEpochs, ...
    'MiniBatchSize',64, ...
    'Shuffle','every-epoch', ...
    'Verbose',true);

trainingData = pixelLabelImageDatastore(imds, pxds);

nameNet = fullfile(dataSaveDir, ...
    sprintf('BestAttentionUNet_Case_%d_Enc_%s_numL_%s_NumEpochs_%d.mat', ...
    currentCase, nameEncoder, nameLayers, numEpochs));
disp(['Training (best Attention U-Net): ' nameNet]);

%% --------- TIME THE TRAINING ----------
tTrainStart = tic;
net = trainNetwork(trainingData, lgraph, opts);
trainTime = toc(tTrainStart);
fprintf('Training time (best Attention U-Net) = %.2f seconds\n', trainTime);

save(nameNet,'net');

%% --------- TIME THE INFERENCE (on composite) ----------
tInferStart = tic;
C = semanticseg(uint8(dataRanden{currentCase}), net);
inferTime = toc(tInferStart);
fprintf('Inference time (composite, best Attention U-Net) = %.4f seconds\n', inferTime);

%% Overlay for visualisation
B = labeloverlay(uint8(dataRanden{currentCase}), C);
figure(1);
imagesc(B); colormap gray;
title('Attention U-Net (best config) - overlay on composite');

% Save overlay image
imwrite(B, fullfile(dataSaveDir, 'attention_unet_overlay.png'));

%% Convert categorical C -> numeric result
result = zeros(size(C), 'like', maskRanden{currentCase});
for k = 1:numClasses
    classMask = (C == classNames(k));
    result(classMask) = k;
end

%% Correctness map (white = correct, black = incorrect)
correctMap = uint8(result == maskRanden{currentCase}) * 255;

figure(2);
imagesc(correctMap); colormap gray;
title('Attention U-Net (best config) - correctly classified pixels (white)');

% Save correctness image
imwrite(correctMap, fullfile(dataSaveDir, 'attention_best_correct.png'));

%% Pixel-wise accuracy
accVal = sum(sum(result == maskRanden{currentCase})) / (rows*cols);

fprintf('Best Attention U-Net Accuracy (Case=%d, Layers=%s, Opt=%s, Epochs=%d) = %.4f\n', ...
    currentCase, nameLayers, typeEncoder, numEpochs, accVal);

%% --------- Additional metrics: Precision, Recall, F1 ----------
gtMask = maskRanden{currentCase};

precision_all = zeros(numClasses,1);
recall_all    = zeros(numClasses,1);
f1_all        = zeros(numClasses,1);
tp_all        = zeros(numClasses,1);
tn_all        = zeros(numClasses,1);
fp_all        = zeros(numClasses,1);
fn_all        = zeros(numClasses,1);

for k = 1:numClasses
    TP = sum((result(:) == k) & (gtMask(:) == k));
    FP = sum((result(:) == k) & (gtMask(:) ~= k));
    FN = sum((result(:) ~= k) & (gtMask(:) == k));
    TN = sum((result(:) ~= k) & (gtMask(:) ~= k));

    tp_all(k) = TP;
    tn_all(k) = TN;
    fp_all(k) = FP;
    fn_all(k) = FN;

    if (TP + FP) > 0
        precision_all(k) = TP / (TP + FP);
    else
        precision_all(k) = 0;
    end

    if (TP + FN) > 0
        recall_all(k) = TP / (TP + FN);
    else
        recall_all(k) = 0;
    end

    if (precision_all(k) + recall_all(k)) > 0
        f1_all(k) = 2 * precision_all(k) * recall_all(k) / (precision_all(k) + recall_all(k));
    else
        f1_all(k) = 0;
    end
end

meanPrecision = mean(precision_all);
meanRecall    = mean(recall_all);
meanF1        = mean(f1_all);

fprintf('Mean Precision (best Attention U-Net) = %.4f\n', meanPrecision);
fprintf('Mean Recall    (best Attention U-Net) = %.4f\n', meanRecall);
fprintf('Mean F1-score  (best Attention U-Net) = %.4f\n', meanF1);

%% Save per-class metrics
metricsTable_attn = table((1:numClasses)', tp_all, tn_all, fp_all, fn_all, ...
    precision_all, recall_all, f1_all, ...
    'VariableNames', {'Class','TP','TN','FP','FN','Precision','Recall','F1Score'});

writetable(metricsTable_attn, fullfile(dataSaveDir, 'best_attention_unet_per_class_metrics.csv'));

%% Save summary results
results_best_attn = table( ...
    currentCase, numLayersNet, {typeEncoder}, numEpochs, accVal, ...
    meanPrecision, meanRecall, meanF1, trainTime, inferTime, ...
    'VariableNames', {'Case','Layers','Optimizer','Epochs','Accuracy', ...
    'MeanPrecision','MeanRecall','MeanF1Score','TrainTime_s','InferTime_s'});

save(fullfile(dataSaveDir,'best_attention_unet_results.mat'),'results_best_attn');
writetable(results_best_attn, fullfile(dataSaveDir,'best_attention_unet_results.csv'));

disp('Saved best Attention U-Net timing and metrics results.');

figure(3);
imagesc(dataRanden{1});
colormap gray;
title('Composite image dataRanden{1}');