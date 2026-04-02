%% segmentationTextureUnet1.m
% Train U-Net on my 9 petro textures (Case 1) and evaluate on dataRanden{1}.

clc;

%% Paths (Windows) 
cd('C:\SRP\TextureSegmentation\Texture-Segmentation-master\CODE');
dataSetDir  = 'C:\SRP\TextureSegmentation\Texture-Segmentation-master\CODE\';
dataSaveDir = 'C:\SRP\TextureSegmentation\Texture-Segmentation-master\CODE\Results\';

if ~exist(dataSaveDir,'dir'), mkdir(dataSaveDir); end

%% Load randenData and apply texture changes
load randenData
clear resRanden stdsRanden meansRanden fname ind edge error*

% Replace Randen textures in Case 1 by my 9 petro textures
changetrainRanden1      

% Build a new composite dataRanden{1} using maskRanden{1} + my textures
changeDataRanden1       

% Pre-allocate accuracy array (layers, case, encoder, epochIndex)
accuracy(3,9,3,4) = 0;

results = [];

%% Loop for training and segmentation (only Case 1 for now)
for currentCase = 1

    [rows,cols] = size(dataRanden{currentCase});
    numClasses  = size(trainRanden{currentCase},3);

    imageDir = fullfile(dataSetDir, 'trainingImages', sprintf('Case_%d',currentCase));
    labelDir = fullfile(dataSetDir, 'trainingLabels', sprintf('Case_%d',currentCase));

    imds  = imageDatastore(imageDir);
    imds2 = imageDatastore(labelDir); %#ok<NASGU> % not used directly, kept for reference

    % Define class names "T1","T2",... matching the labels
    clear classNames
    for counterClass = 1:numClasses
        classNames(counterClass) = strcat("T", num2str(counterClass));
    end

    labelIDs = 1:numClasses;
    pxds = pixelLabelDatastore(labelDir, classNames, labelIDs);

    % === choose which epochs to run ===
    for numEpochsName = 1:4  % change to 1:4 
        switch numEpochsName
            case 1
                numEpochs = 10;
            case 2
                numEpochs = 24;
            case 3
                numEpochs = 50;
            case 4
                numEpochs = 100;
        end

        % === choose encoder (optimizer): 1=sgdm, 2=adam, 3=rmsprop ===
        for caseEncoder = 1:3  % 2 = Adam 
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

            % === choose depth: 1=15-layer, 2=20-layer, 3=20B ===
            numFilters = 64;
            filterSize = 3;

            for numLayersNetwork = 1:3  % set to 2 or 3 for deeper versions
                switch numLayersNetwork
                    case 1  % 15-layer net (paper)
                        layers = [
                            imageInputLayer([32 32 1])
                            convolution2dLayer(filterSize,numFilters,'Padding',1)
                            reluLayer()
                            maxPooling2dLayer(2,'Stride',2)
                            convolution2dLayer(filterSize,numFilters,'Padding',1)
                            reluLayer()
                            maxPooling2dLayer(2,'Stride',2)
                            convolution2dLayer(filterSize,numFilters,'Padding',1)
                            reluLayer()
                            transposedConv2dLayer(4,numFilters,'Stride',2,'Cropping',1)
                            convolution2dLayer(1,numClasses)
                            transposedConv2dLayer(4,numFilters,'Stride',2,'Cropping',1)
                            convolution2dLayer(1,numClasses)
                            softmaxLayer()
                            pixelClassificationLayer()
                        ];
                        nameLayers = '15';

                    case 2  % 20-layer
                        layers = [
                            imageInputLayer([32 32 1])
                            convolution2dLayer(filterSize,numFilters,'Padding',1)
                            reluLayer()
                            maxPooling2dLayer(2,'Stride',2)
                            convolution2dLayer(filterSize,numFilters,'Padding',1)
                            reluLayer()
                            maxPooling2dLayer(2,'Stride',2)
                            convolution2dLayer(filterSize,numFilters,'Padding',1)
                            reluLayer()
                            maxPooling2dLayer(2,'Stride',2)
                            convolution2dLayer(filterSize,numFilters,'Padding',1)
                            reluLayer()
                            transposedConv2dLayer(4,numFilters,'Stride',2,'Cropping',1)
                            convolution2dLayer(1,numClasses)
                            transposedConv2dLayer(4,numFilters,'Stride',2,'Cropping',1)
                            convolution2dLayer(1,numClasses)
                            transposedConv2dLayer(4,numFilters,'Stride',2,'Cropping',1)
                            convolution2dLayer(1,numClasses)
                            softmaxLayer()
                            pixelClassificationLayer()
                        ];
                        nameLayers = '20';

                    case 3  % 20B 
                        layers = [
                            imageInputLayer([32 32 1])
                            convolution2dLayer(filterSize,numFilters,'Padding',1)
                            reluLayer()
                            maxPooling2dLayer(2,'Stride',2)
                            convolution2dLayer(filterSize,numFilters,'Padding',1)
                            reluLayer()
                            maxPooling2dLayer(2,'Stride',2)
                            convolution2dLayer(filterSize,numFilters,'Padding',1)
                            reluLayer()
                            maxPooling2dLayer(2,'Stride',2)
                            convolution2dLayer(filterSize,numFilters,'Padding',1)
                            reluLayer()
                            transposedConv2dLayer(4,numFilters,'Stride',2,'Cropping',1)
                            convolution2dLayer(1,numClasses)
                            transposedConv2dLayer(4,numFilters,'Stride',2,'Cropping',1)
                            convolution2dLayer(1,numClasses)
                            transposedConv2dLayer(4,numFilters,'Stride',2,'Cropping',1)
                            convolution2dLayer(1,numClasses)
                            softmaxLayer()
                            pixelClassificationLayer()
                        ];
                        nameLayers = '20B';
                end

                % --- Training options ---
                opts = trainingOptions(typeEncoder, ...
                    'InitialLearnRate',1e-3, ...
                    'MaxEpochs',numEpochs, ...
                    'MiniBatchSize',64, ...
                    'Shuffle','every-epoch', ...
                    'Verbose',true);

                trainingData = pixelLabelImageDatastore(imds, pxds);
                nameNet = fullfile(dataSaveDir, ...
                    sprintf('Network_Case_%d_Enc_%s_numL_%s_NumEpochs_%d.mat', ...
                    currentCase, nameEncoder, nameLayers, numEpochs));
                disp(['Training: ' nameNet]);

                net = trainNetwork(trainingData, layers, opts);

                % ==== Evaluate on composite dataRanden{1} ====
                C = semanticseg(uint8(dataRanden{currentCase}), net);

                % Overlay for visualisation
                B = labeloverlay(uint8(dataRanden{currentCase}), C);
                figure(currentCase+1);
                imagesc(B); colormap gray;
                title('Overlay of prediction on composite');

                % Convert categorical C -> numeric result
                result = zeros(size(C), 'like', maskRanden{currentCase});
                for k = 1:numClasses
                    classMask = (C == classNames(k));   % logical mask for class k
                    result(classMask) = k;              % assign label k
                end

                figure(currentCase);
                imagesc(result == maskRanden{currentCase});
                title('Correctly classified pixels (white)');

                % Pixel-wise accuracy
                accVal = sum(sum(result == maskRanden{currentCase})) / (rows*cols);
                accuracy(numLayersNetwork,currentCase,caseEncoder,numEpochsName) = accVal;

                fprintf('Accuracy (Case=%d, Layers=%s, Opt=%s, Epochs=%d) = %.4f\n', ...
                    currentCase, nameLayers, typeEncoder, numEpochs, accVal);

                results = [results;
                    currentCase, str2double(nameLayers(1:2)), caseEncoder, numEpochs, accVal];

                % save network and running accuracy
                save(nameNet,'net');
                save(fullfile(dataSaveDir, 'accuracy.mat'),'accuracy','results');
            end
        end
    end
end

% Misclassification (%) array (same indexing as accuracy)
misclassification = 100*(1 - accuracy);

% Save results to LaTeX table
if ~isempty(results)

    % Create table
    T = array2table(results, ...
        'VariableNames', {'Case','Layers','EncoderIndex','Epochs','Accuracy'});

    latexFile = fullfile(dataSaveDir,'segmentation_unet_case1_results.tex');
    fid = fopen(latexFile,'w');

    fprintf(fid,'\\begin{table}[h]\n');
    fprintf(fid,'\\centering\n');
    fprintf(fid,'\\caption{Segmentation Results -- Case 1}\n');
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

    disp('Saved LaTeX table to segmentation_unet_case1_results.tex');
end

figure;
imagesc(dataRanden{1});
colormap gray;
title('Composite image dataRanden{1}');
