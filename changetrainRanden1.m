% changetrainRanden1.m
% Use the 9 texture images 1.jpg..9.jpg to replace trainRanden{1}
% made to have 10 texture images, but I was only given 9 

% load randenData

clc

u1  = imread('1.jpg');
u2  = imread('2.jpg');
u3  = imread('3.jpg');
u4  = imread('4.jpg');
u5  = imread('5.jpg');
u6  = imread('6.jpg');
u7  = imread('7.jpg');
u8  = imread('8.jpg');
u9  = imread('9.jpg');

% If they are colour, convert to grayscale
if size(u1,3)==3,  u1 = rgb2gray(u1); end
if size(u2,3)==3,  u2 = rgb2gray(u2); end
if size(u3,3)==3,  u3 = rgb2gray(u3); end
if size(u4,3)==3,  u4 = rgb2gray(u4); end
if size(u5,3)==3,  u5 = rgb2gray(u5); end
if size(u6,3)==3,  u6 = rgb2gray(u6); end
if size(u7,3)==3,  u7 = rgb2gray(u7); end
if size(u8,3)==3,  u8 = rgb2gray(u8); end
if size(u9,3)==3,  u9 = rgb2gray(u9); end

% Copy original cell array
NtrainRanden = trainRanden;

% Stack the 9 textures along the 3rd dimension
NewtrainRanden(:,:,1) = u1;
NewtrainRanden(:,:,2) = u2;
NewtrainRanden(:,:,3) = u3;
NewtrainRanden(:,:,4) = u4;
NewtrainRanden(:,:,5) = u5;
NewtrainRanden(:,:,6) = u6;
NewtrainRanden(:,:,7) = u7;
NewtrainRanden(:,:,8) = u8;
NewtrainRanden(:,:,9) = u9;
% NOTE: no u10 here because we only have 9 images

NtrainRanden{1} = NewtrainRanden;
trainRanden{1}  = NtrainRanden{1};

fprintf('changetrainRanden1: trainRanden{1} now has %d classes (textures).\n', size(trainRanden{1},3));
