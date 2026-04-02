%% changeDataRanden1.m
% It uses maskRanden{1} and replaces each class region by one of my
% petrographic textures (1.jpg ... 9.jpg).

% load randenData so maskRanden is in the workspace.

clc

if ~exist('maskRanden','var')
    error('maskRanden not found. Run "load randenData" before changeDataRanden1.');
end

mask = maskRanden{1};
[rows, cols] = size(mask);

%% --- Read my texture images ---

u1 = imread('1.jpg');
u2 = imread('2.jpg');
u3 = imread('3.jpg');
u4 = imread('4.jpg');
u5 = imread('5.jpg');
u6 = imread('6.jpg');
u7 = imread('7.jpg');
u8 = imread('8.jpg');
u9 = imread('9.jpg');

% Convert to grayscale if needed, and resize to match the mask (256×256).
tex = cell(1,9);
U   = {u1,u2,u3,u4,u5,u6,u7,u8,u9};
for k = 1:9
    Ik = U{k};

    % If RGB, convert to gray
    if ndims(Ik) == 3
        Ik = rgb2gray(Ik);
    end

    % Resize if not already same size as mask
    if ~isequal(size(Ik), [rows cols])
        Ik = imresize(Ik, [rows cols]);
    end

    tex{k} = double(Ik);   % store as double for multiplication with masks
end

u1 = tex{1};
u2 = tex{2};
u3 = tex{3};
u4 = tex{4};
u5 = tex{5};
u6 = tex{6};
u7 = tex{7};
u8 = tex{8};
u9 = tex{9};
% If later I get a 10th texture, I can add u10 here like in Dr's code.

%% --- 

dataRanden{9} = zeros(rows,cols);  

mask1  = double(mask == 1);  class1  = mask1  .* u1;
mask2  = double(mask == 2);  class2  = mask2  .* u2;
mask3  = double(mask == 3);  class3  = mask3  .* u3;
mask4  = double(mask == 4);  class4  = mask4  .* u4;
mask5  = double(mask == 5);  class5  = mask5  .* u5;
mask6  = double(mask == 6);  class6  = mask6  .* u6;
mask7  = double(mask == 7);  class7  = mask7  .* u7;
mask8  = double(mask == 8);  class8  = mask8  .* u8;
mask9  = double(mask == 9);  class9  = mask9  .* u9;


% Composite = sum of class images
dataRanden{1} = class1 + class2 + class3 + class4 + ...
                class5 + class6 + class7 + class8 + class9;

%% --- Visualise the composite exactly like Dr, but with fixed contrast ---
figure;
imagesc(dataRanden{1}, [0 255]);  % fix range so it’s not all white
colormap gray;
axis image off;
title('dataRanden1 composite from my 9 textures');

fprintf('Composite range: min = %.1f, max = %.1f\n', ...
    min(dataRanden{1}(:)), max(dataRanden{1}(:)));
