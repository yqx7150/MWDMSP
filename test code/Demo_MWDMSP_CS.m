close all;
clear;
clc;

getd('./');
% set to 0 if you want to run on CPU (very slow)
gpu = 1;
if gpu 
    gpuDevice(gpu); 
end

%% CS demo
%#######%%%%% read sampling %%%%
% line = 43  
% [mask] = strucrand(256,256,1,line);
load mask_random015; mask = mask_random015;
figure(355); imshow(mask,[]);   %
figure(356); imshow(fftshift(mask),[]);  
n = size(mask,2);
fprintf(1, 'n=%d, k=%d, Unsamped=%f\n', n, sum(sum(mask)),1-sum(sum(mask))/n/n); %

%% load image
gt = double(imread('images/lena.tif'));
% gt = double(imread('images/boats.tif'));
% gt = double(imread('images/cameraman.tif'));
% gt = double(imread('images/baboon.tif'));
% gt = double(imread('images/peppers.tif'));
% gt = double(imread('images/straw.tif'));

figure(334);imshow(abs(gt),[]);

sigma_d = 0 * 255;
noise = randn(size(gt));
degraded = mask.*(fft2(gt) + noise * sigma_d + (0+1i)*noise * sigma_d); %
Im = ifft2(degraded); 
figure(22); imshow(Im,[],'Border','tight');

params.gt = gt;

% run DAEP
params.sigma_net = 5;   %20;  %15;  %11;  %
params.num_iter = 600;

load('E:\TDMSP_ÑµÁ·Ñ¹Ëõ¸ÐÖª\2019.4.9\data\MWCNN_GDSigma5_3_3Dgrey_400_input_threewavelet\MWCNN_GDSigma5_3_3Dgrey_400_input_threewavelet-epoch-44');  net1 = net;
net1 = dagnn.DagNN.loadobj(net1) ;
net1.removeLayer('objective') ;
out_idx = net1.getVarIndex('prediction') ;
net1.vars(net1.getVarIndex('prediction')).precious = 1 ;
net1.mode = 'test';
if gpu
    net1.move('gpu'); 
end

params.out_idx = out_idx;  params.gpu = gpu;  
params2.out_idx = out_idx;  params2.gpu = gpu;  
params3.out_idx = out_idx;  params3.gpu = gpu;  
[map_deblur,psnr_psnr] = Complex_DMSPMRIRec_natureimage_1sigma(Im, degraded, mask, sigma_d, params,net1);

[psnr4, ssim4, fsim4, ergas4, sam4] = MSIQA(abs(gt), abs(map_deblur));
[psnr4, ssim4, fsim4, ergas4, sam4]

figure(666);
subplot(2,3,[4,5,6]);imshow([abs(Im-gt)/255,abs(map_deblur-gt)/255],[]); title('Recon-error');colormap(jet);colorbar;
subplot(2,3,1);imshow(abs(gt)/255); title('Ground-truth');colormap(gray);
subplot(2,3,2);imshow(abs(Im)/255); title('Zero-filled');colormap(gray);
subplot(2,3,3);imshow(abs(map_deblur)/255); title('Net-recon');colormap(gray);
figure(667);imshow([real(gt)/255,imag(gt)/255,abs(gt)/255],[]); 
figure(668);imshow([abs(Im-gt)/255,abs(map_deblur-gt)/255],[]); colormap(jet);colorbar;



