%% attentionUnet20layers.m
function lgraph = attentionUnet20layers(numClasses)
% Deeper Attention U-Net (3 encoder levels, full decoder + attention gates)
% 32x32 input, grayscale
% Channels: 64 -> 128 -> 256 -> 512 (bottleneck)

input = imageInputLayer([32 32 1],'Name','input');

%% Encoder 1 (32x32 -> 16x16)
enc1 = [
    convolution2dLayer(3,64,'Padding','same','Name','enc1_conv1')
    reluLayer('Name','enc1_relu1')
    convolution2dLayer(3,64,'Padding','same','Name','enc1_conv2')
    reluLayer('Name','enc1_relu2')
];
pool1 = maxPooling2dLayer(2,'Stride',2,'Name','pool1');

%% Encoder 2 (16x16 -> 8x8)
enc2 = [
    convolution2dLayer(3,128,'Padding','same','Name','enc2_conv1')
    reluLayer('Name','enc2_relu1')
    convolution2dLayer(3,128,'Padding','same','Name','enc2_conv2')
    reluLayer('Name','enc2_relu2')
];
pool2 = maxPooling2dLayer(2,'Stride',2,'Name','pool2');

%% Encoder 3 (8x8 -> 4x4)
enc3 = [
    convolution2dLayer(3,256,'Padding','same','Name','enc3_conv1')
    reluLayer('Name','enc3_relu1')
    convolution2dLayer(3,256,'Padding','same','Name','enc3_conv2')
    reluLayer('Name','enc3_relu2')
];
pool3 = maxPooling2dLayer(2,'Stride',2,'Name','pool3');

%% Bottleneck (4x4)
bottleneck = [
    convolution2dLayer(3,512,'Padding','same','Name','bottleneck_conv1')
    reluLayer('Name','bottleneck_relu1')
    convolution2dLayer(3,512,'Padding','same','Name','bottleneck_conv2')
    reluLayer('Name','bottleneck_relu2')
];

%% Decoder (upsample + conv blocks)
% up1 : bottleneck (4x4) -> dec3 (8x8)  (matches enc3)
up1 = transposedConv2dLayer(2,256,'Stride',2,'Name','up1');
dec3 = [
    convolution2dLayer(3,256,'Padding','same','Name','dec3_conv1')
    reluLayer('Name','dec3_relu1')
    convolution2dLayer(3,256,'Padding','same','Name','dec3_conv2')
    reluLayer('Name','dec3_relu2')
];

% up2 : dec3 (8x8) -> dec2 (16x16) (matches enc2)
up2 = transposedConv2dLayer(2,128,'Stride',2,'Name','up2');
dec2 = [
    convolution2dLayer(3,128,'Padding','same','Name','dec2_conv1')
    reluLayer('Name','dec2_relu1')
    convolution2dLayer(3,128,'Padding','same','Name','dec2_conv2')
    reluLayer('Name','dec2_relu2')
];

% up3 : dec2 (16x16) -> dec1 (32x32) (matches enc1)
up3 = transposedConv2dLayer(2,64,'Stride',2,'Name','up3');
dec1 = [
    convolution2dLayer(3,64,'Padding','same','Name','dec1_conv1')
    reluLayer('Name','dec1_relu1')
    convolution2dLayer(3,64,'Padding','same','Name','dec1_conv2')
    reluLayer('Name','dec1_relu2')
];

%% Final layers
finalLayers = [
    convolution2dLayer(1,numClasses,'Name','final_conv')
    softmaxLayer('Name','softmax')
    pixelClassificationLayer('Name','pixelClass')
];

%% Build graph
lgraph = layerGraph;

% Add encoder, bottleneck, decoder, final
lgraph = addLayers(lgraph,input);
lgraph = addLayers(lgraph,enc1);
lgraph = addLayers(lgraph,pool1);
lgraph = addLayers(lgraph,enc2);
lgraph = addLayers(lgraph,pool2);
lgraph = addLayers(lgraph,enc3);
lgraph = addLayers(lgraph,pool3);
lgraph = addLayers(lgraph,bottleneck);

lgraph = addLayers(lgraph,up1);
lgraph = addLayers(lgraph,dec3);
lgraph = addLayers(lgraph,up2);
lgraph = addLayers(lgraph,dec2);
lgraph = addLayers(lgraph,up3);
lgraph = addLayers(lgraph,dec1);
lgraph = addLayers(lgraph,finalLayers);

%% Encoder connections
lgraph = connectLayers(lgraph,'input','enc1_conv1');
lgraph = connectLayers(lgraph,'enc1_relu2','pool1');
lgraph = connectLayers(lgraph,'pool1','enc2_conv1');
lgraph = connectLayers(lgraph,'enc2_relu2','pool2');
lgraph = connectLayers(lgraph,'pool2','enc3_conv1');
lgraph = connectLayers(lgraph,'enc3_relu2','pool3');
lgraph = connectLayers(lgraph,'pool3','bottleneck_conv1');

%% Decoder + Attention Gate stage 1 (bottleneck -> up1 -> concat with enc3)
% connect bottleneck to up1
lgraph = connectLayers(lgraph,'bottleneck_relu2','up1');

% attention gate for enc3 skip (skip: enc3_relu2, gating: up1)
lgraph = addAttentionGate(lgraph,'enc3_relu2','up1','att1');

% concat and connect to dec3
concat3 = depthConcatenationLayer(2,'Name','concat3');
lgraph = addLayers(lgraph,concat3);
lgraph = connectLayers(lgraph,'att1_mult','concat3/in1');
lgraph = connectLayers(lgraph,'up1','concat3/in2');
lgraph = connectLayers(lgraph,'concat3','dec3_conv1');

%% Decoder stage 2 (dec3 -> up2 -> concat with enc2)
lgraph = connectLayers(lgraph,'dec3_relu2','up2');

% attention gate for enc2 skip (skip: enc2_relu2, gating: up2)
lgraph = addAttentionGate(lgraph,'enc2_relu2','up2','att2');

concat2 = depthConcatenationLayer(2,'Name','concat2');
lgraph = addLayers(lgraph,concat2);
lgraph = connectLayers(lgraph,'att2_mult','concat2/in1');
lgraph = connectLayers(lgraph,'up2','concat2/in2');
lgraph = connectLayers(lgraph,'concat2','dec2_conv1');

%% Decoder stage 3 (dec2 -> up3 -> concat with enc1)
lgraph = connectLayers(lgraph,'dec2_relu2','up3');

% attention gate for enc1 skip (skip: enc1_relu2, gating: up3)
lgraph = addAttentionGate(lgraph,'enc1_relu2','up3','att3');

concat1 = depthConcatenationLayer(2,'Name','concat1');
lgraph = addLayers(lgraph,concat1);
lgraph = connectLayers(lgraph,'att3_mult','concat1/in1');
lgraph = connectLayers(lgraph,'up3','concat1/in2');
lgraph = connectLayers(lgraph,'concat1','dec1_conv1');

%% Final connection
lgraph = connectLayers(lgraph,'dec1_relu2','final_conv');

end