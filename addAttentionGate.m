%% addAttentionGate.m
function lgraph = addAttentionGate(lgraph, skipLayer, gatingLayer, prefix)

theta = convolution2dLayer(1,64,'Padding','same','Name',[prefix '_theta']);
phi   = convolution2dLayer(1,64,'Padding','same','Name',[prefix '_phi']);
add   = additionLayer(2,'Name',[prefix '_add']);
relu  = reluLayer('Name',[prefix '_relu']);
psi   = convolution2dLayer(1,1,'Padding','same','Name',[prefix '_psi']);
sig   = sigmoidLayer('Name',[prefix '_sig']);
mult  = multiplicationLayer(2,'Name',[prefix '_mult']);

lgraph = addLayers(lgraph,theta);
lgraph = addLayers(lgraph,phi);
lgraph = addLayers(lgraph,add);
lgraph = addLayers(lgraph,relu);
lgraph = addLayers(lgraph,psi);
lgraph = addLayers(lgraph,sig);
lgraph = addLayers(lgraph,mult);

lgraph = connectLayers(lgraph,skipLayer,[prefix '_theta']);
lgraph = connectLayers(lgraph,gatingLayer,[prefix '_phi']);

lgraph = connectLayers(lgraph,[prefix '_theta'],[prefix '_add/in1']);
lgraph = connectLayers(lgraph,[prefix '_phi'],[prefix '_add/in2']);

lgraph = connectLayers(lgraph,[prefix '_add'],[prefix '_relu']);
lgraph = connectLayers(lgraph,[prefix '_relu'],[prefix '_psi']);
lgraph = connectLayers(lgraph,[prefix '_psi'],[prefix '_sig']);

lgraph = connectLayers(lgraph,skipLayer,[prefix '_mult/in1']);
lgraph = connectLayers(lgraph,[prefix '_sig'],[prefix '_mult/in2']);

end