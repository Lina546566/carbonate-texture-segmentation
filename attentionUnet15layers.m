%% attentionUnet15layers.m 
function lgraph = attentionUnet15layers(numClasses)

% Proper Attention U-Net (15-layer style)
% 32x32 input, grayscale
% Includes attention gates on skip connections

input = imageInputLayer([32 32 1],'Name','input');

%% Encoder Block 1
enc1 = [
    convolution2dLayer(3,64,'Padding','same','Name','enc1_conv1')
    reluLayer('Name','enc1_relu1')
    convolution2dLayer(3,64,'Padding','same','Name','enc1_conv2')
    reluLayer('Name','enc1_relu2')
];
pool1 = maxPooling2dLayer(2,'Stride',2,'Name','pool1');

%% Encoder Block 2
enc2 = [
    convolution2dLayer(3,128,'Padding','same','Name','enc2_conv1')
    reluLayer('Name','enc2_relu1')
    convolution2dLayer(3,128,'Padding','same','Name','enc2_conv2')
    reluLayer('Name','enc2_relu2')
];
pool2 = maxPooling2dLayer(2,'Stride',2,'Name','pool2');

%% Bottleneck
bottleneck = [
    convolution2dLayer(3,256,'Padding','same','Name','bottleneck_conv1')
    reluLayer('Name','bottleneck_relu1')
    convolution2dLayer(3,256,'Padding','same','Name','bottleneck_conv2')
    reluLayer('Name','bottleneck_relu2')
];

%% Decoder
up1 = transposedConv2dLayer(2,128,'Stride',2,'Name','up1');
up2 = transposedConv2dLayer(2,64,'Stride',2,'Name','up2');

dec1 = [
    convolution2dLayer(3,128,'Padding','same','Name','dec1_conv1')
    reluLayer('Name','dec1_relu1')
    convolution2dLayer(3,128,'Padding','same','Name','dec1_conv2')
    reluLayer('Name','dec1_relu2')
];

dec2 = [
    convolution2dLayer(3,64,'Padding','same','Name','dec2_conv1')
    reluLayer('Name','dec2_relu1')
    convolution2dLayer(3,64,'Padding','same','Name','dec2_conv2')
    reluLayer('Name','dec2_relu2')
];

finalLayers = [
    convolution2dLayer(1,numClasses,'Name','final_conv')
    softmaxLayer('Name','softmax')
    pixelClassificationLayer('Name','pixelClass')
];

%% Build Graph
lgraph = layerGraph;

lgraph = addLayers(lgraph,input);
lgraph = addLayers(lgraph,enc1);
lgraph = addLayers(lgraph,pool1);
lgraph = addLayers(lgraph,enc2);
lgraph = addLayers(lgraph,pool2);
lgraph = addLayers(lgraph,bottleneck);
lgraph = addLayers(lgraph,up1);
lgraph = addLayers(lgraph,dec1);
lgraph = addLayers(lgraph,up2);
lgraph = addLayers(lgraph,dec2);
lgraph = addLayers(lgraph,finalLayers);

%% Connect Encoder
lgraph = connectLayers(lgraph,'input','enc1_conv1');
lgraph = connectLayers(lgraph,'enc1_relu2','pool1');
lgraph = connectLayers(lgraph,'pool1','enc2_conv1');
lgraph = connectLayers(lgraph,'enc2_relu2','pool2');
lgraph = connectLayers(lgraph,'pool2','bottleneck_conv1');

%% Attention Gate 1 (enc2 skip)
lgraph = addAttentionGate(lgraph,'enc2_relu2','up1','att1');

%% Decoder Connections
lgraph = connectLayers(lgraph,'bottleneck_relu2','up1');
concat1 = depthConcatenationLayer(2,'Name','concat1');
lgraph = addLayers(lgraph,concat1);
lgraph = connectLayers(lgraph,'att1_mult','concat1/in1');
lgraph = connectLayers(lgraph,'up1','concat1/in2');
lgraph = connectLayers(lgraph,'concat1','dec1_conv1');

%% Attention Gate 2 (enc1 skip)
lgraph = connectLayers(lgraph,'dec1_relu2','up2');
lgraph = addAttentionGate(lgraph,'enc1_relu2','up2','att2');

concat2 = depthConcatenationLayer(2,'Name','concat2');
lgraph = addLayers(lgraph,concat2);
lgraph = connectLayers(lgraph,'att2_mult','concat2/in1');
lgraph = connectLayers(lgraph,'up2','concat2/in2');
lgraph = connectLayers(lgraph,'concat2','dec2_conv1');

lgraph = connectLayers(lgraph,'dec2_relu2','final_conv');

end