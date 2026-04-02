%% segmentationTextureAttentionUnet1.m
% Train Attention U-Net on my 9 petro textures (Case 1)
% and evaluate on dataRanden{1}

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

accuracy(3,9,3,4) = 0;
results = [];

%% Loop for training and segmentation
for currentCase = 1

    [rows,cols] = size(dataRanden{currentCase});
    numClasses  = size(trainRanden{currentCase},3);

    imageDir = fullfile(dataSetDir, 'trainingImages', sprintf('Case_%d',currentCase));
    labelDir = fullfile(dataSetDir, 'trainingLabels', sprintf('Case_%d',currentCase));

    imds  = imageDatastore(imageDir);

    clear classNames
    for counterClass = 1:numClasses
        classNames(counterClass) = strcat("T", num2str(counterClass));
    end

    labelIDs = 1:numClasses;
    pxds = pixelLabelDatastore(labelDir, classNames, labelIDs);

    %% Epoch loop
    for numEpochsName = 1:4

        switch numEpochsName
            case 1, numEpochs = 10;
            case 2, numEpochs = 24;
            case 3, numEpochs = 50;
            case 4, numEpochs = 100;
        end

        %% Optimizer loop
        for caseEncoder = 1:3

            switch caseEncoder
                case 1
                    typeEncoder = 'sgdm';
                    nameEncoder = '1';
                case 2
                    typeEncoder = 'adam';
                    nameEncoder = '2';
                case 3
                    typeEncoder = 'rmsprop';
                    nameEncoder = '3';
            end

            %% Network depth loop
            for numLayersNetwork = 1:2  % 1=15layer, 2=20layer

                switch numLayersNetwork
                    case 1
                        lgraph = attentionUnet15layers(numClasses);
                        nameLayers = '15';
                    case 2
                        lgraph = attentionUnet20layers(numClasses);
                        nameLayers = '20';
                end

                %% Training options
                opts = trainingOptions(typeEncoder, ...
                    'InitialLearnRate',1e-3, ...
                    'MaxEpochs',numEpochs, ...
                    'MiniBatchSize',64, ...
                    'Shuffle','every-epoch', ...
                    'Verbose',true);

                trainingData = pixelLabelImageDatastore(imds, pxds);

                nameNet = fullfile(dataSaveDir, ...
                    sprintf('AttentionUNet_Case_%d_Enc_%s_numL_%s_NumEpochs_%d.mat', ...
                    currentCase, nameEncoder, nameLayers, numEpochs));

                disp(['Training: ' nameNet]);

                net = trainNetwork(trainingData, lgraph, opts);

                %% ==== Evaluate on composite image ====
                C = semanticseg(uint8(dataRanden{currentCase}), net);

                B = labeloverlay(uint8(dataRanden{currentCase}), C);
                figure(currentCase+1);
                imagesc(B); colormap gray;
                title('Attention U-Net Overlay');

                result = zeros(size(C), 'like', maskRanden{currentCase});
                for k = 1:numClasses
                    classMask = (C == classNames(k));
                    result(classMask) = k;
                end

                figure(currentCase);
                imagesc(result == maskRanden{currentCase});
                title('Correctly classified pixels');

                accVal = sum(sum(result == maskRanden{currentCase})) / (rows*cols);

                accuracy(numLayersNetwork,currentCase,caseEncoder,numEpochsName) = accVal;

                fprintf('Attention U-Net Accuracy (Case=%d, Layers=%s, Opt=%s, Epochs=%d) = %.4f\n', ...
                    currentCase, nameLayers, typeEncoder, numEpochs, accVal);

                results = [results;
                    currentCase, str2double(nameLayers), caseEncoder, numEpochs, accVal];

                save(nameNet,'net');
                save(fullfile(dataSaveDir, 'accuracy_attention.mat'),'accuracy','results');

            end
        end
    end
end

misclassification = 100*(1 - accuracy);

%% Save LaTeX table
if ~isempty(results)

    T = array2table(results, ...
        'VariableNames', {'Case','Layers','EncoderIndex','Epochs','Accuracy'});

    latexFile = fullfile(dataSaveDir,'segmentation_attention_unet_case1_results.tex');
    fid = fopen(latexFile,'w');

    fprintf(fid,'\\begin{table}[h]\n');
    fprintf(fid,'\\centering\n');
    fprintf(fid,'\\caption{Segmentation Results -- Attention U-Net (Case 1)}\n');
    fprintf(fid,'\\begin{tabular}{ccccc}\n');
    fprintf(fid,'\\hline\n');
    fprintf(fid,'Case & Layers & Encoder & Epochs & Accuracy \\\\\n');
    fprintf(fid,'\\hline\n');

    for i = 1:height(T)
        fprintf(fid,'%d & %d & %d & %d & %.4f \\\\\n', ...
            T.Case(i), ...
            T.Layers(i), ...
            T.EncoderIndex(i), ...
            T.Epochs(i), ...
            T.Accuracy(i));
    end

    fprintf(fid,'\\hline\n');
    fprintf(fid,'\\end{tabular}\n');
    fprintf(fid,'\\end{table}\n');

    fclose(fid);

    disp('Saved LaTeX table for Attention U-Net');
end

figure;
imagesc(dataRanden{1});
colormap gray;
title('Composite image dataRanden{1}');