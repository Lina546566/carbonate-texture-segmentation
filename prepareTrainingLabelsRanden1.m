%% prepareTrainingLabelsRanden1.m
% Create 32x32 training patches (single-class and two-class horizontal pairs)

clc;

%% Go to CODE folder (Windows SRP path)
cd('C:\SRP\TextureSegmentation\Texture-Segmentation-master\CODE');

%% Load randenData and apply  textures
load randenData    % loads dataRanden, trainRanden, maskRanden
changetrainRanden1 % replace trainRanden{1} with your 9 textures

% dataRanden    - cell with composite images
% trainRanden   - cell with training textures per case
% maskRanden    - cell with masks per case

clear resRanden stdsRanden meansRanden fname ind edge error*

%% Partition to create many images to train
% select one of the composite images
currentCase   = 1;
imageSize     = [32 32];
stepOverlap   = 0;

figure(1); colormap gray;

[rows,cols,numClasses] = size(trainRanden{currentCase});
fprintf('Preparing training patches: rows=%d, cols=%d, numClasses=%d\n', rows, cols, numClasses);

% Make sure trainingImages and trainingLabels folders exist
if ~exist('trainingImages','dir'), mkdir('trainingImages'); end
if ~exist('trainingLabels','dir'), mkdir('trainingLabels'); end
if ~exist(fullfile('trainingImages','Case_1'),'dir'), mkdir(fullfile('trainingImages','Case_1')); end
if ~exist(fullfile('trainingLabels','Case_1'),'dir'), mkdir(fullfile('trainingLabels','Case_1')); end

%% Single-class patches
for counterClasses = 1:numClasses
    for counterR = 1:imageSize(1)-stepOverlap : rows-imageSize(1)
        for counterC = 1:imageSize(2)-stepOverlap : cols-imageSize(2)

            currentSection = uint8(trainRanden{currentCase}( ...
                counterR:counterR+imageSize(1)-1, ...
                counterC:counterC+imageSize(2)-1, ...
                counterClasses));

            currentLabel = uint8(ones(32) * counterClasses);

            imagesc(currentSection);
            title(sprintf('Class=%d (%d-%d)', counterClasses, counterR, counterC));
            drawnow;

            fName  = sprintf('Texture_Randen_Class_%d_%d_%d.png', counterClasses, counterR, counterC);
            fNameL = sprintf('Texture_Randen_Label_Class_%d_%d_%d.png', counterClasses, counterR, counterC);

            imwrite(currentSection, fullfile('trainingImages',  sprintf('Case_%d',currentCase), fName));
            imwrite(currentLabel,   fullfile('trainingLabels', sprintf('Case_%d',currentCase), fNameL));

        end
    end
end

h2 = imagesc(currentSection);

%% Two-class horizontal patches
for counterClass_1 = 1:numClasses
    for counterClass_2 = 1:numClasses
        if counterClass_1 ~= counterClass_2
            for counterR = 1:imageSize(1)-stepOverlap : rows-imageSize(1)
                for counterC = 1:imageSize(2)-stepOverlap : cols-imageSize(2)

                    currentSection_1 = uint8(trainRanden{currentCase}( ...
                        counterR:counterR+imageSize(1)-1, ...
                        counterC:counterC+imageSize(2)-1, ...
                        counterClass_1));
                    currentLabel_1   = uint8(ones(32)*counterClass_1);

                    currentSection_2 = uint8(trainRanden{currentCase}( ...
                        counterR:counterR+imageSize(1)-1, ...
                        counterC:counterC+imageSize(2)-1, ...
                        counterClass_2));
                    currentLabel_2   = uint8(ones(32)*counterClass_2);

                    currentSection = [currentSection_1(:,1:16) currentSection_2(:,1:16)];
                    currentLabel   = [currentLabel_1(:,1:16)   currentLabel_2(:,1:16)];

                    h2.CData = currentSection;
                    title(sprintf('Classes=%d/%d (%d-%d)', counterClass_1, counterClass_2, counterR, counterC));
                    drawnow;

                    fName  = sprintf('Texture_Randen_Classes_%d_%dR_%dC_%d.png', counterClass_1, counterClass_2, counterR, counterC);
                    fNameL = sprintf('Texture_Randen_Label_Classes_%d_%dR_%dC_%d.png', counterClass_1, counterClass_2, counterR, counterC);

                    imwrite(currentSection, fullfile('trainingImages',  sprintf('Case_%d',currentCase), fName));
                    imwrite(currentLabel,   fullfile('trainingLabels', sprintf('Case_%d',currentCase), fNameL));

                end
            end
        end
    end
end

disp('Finished prepareTrainingLabelsRanden1 for Case 1.');
