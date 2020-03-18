
% add MatCaffe path
addpath ../mnt/data/siavash/caffe/matlab;
getd = @(p)path(path,p);% Add some directories to the path
getd('DMSP_MRIRec/Matlabcode_TBMDUdemo_v1\');
getd('DMSP_MRIRec/骆老师给的真实模拟数据\');
getd('DMSP_MRIRec/traindata_lsq\');
getd('DMSP_MRIRec/DMSP_diffSigma\');
getd('DMSP_MRIRec/quality_assess\');
getd('./');

% set to 0 if you want to run on CPU (very slow)
gpu = 1;
% if gpu 
%     gpuDevice(gpu); 
% end

%% Deblurring demo
%#######%%%%% read sampling %%%%
% line = 43  
% [mask] = strucrand(256,256,1,line);
load mask_random015; mask = mask_random015;
figure(355); imshow(mask,[]);   %
figure(356); imshow(fftshift(mask),[]);  
n = size(mask,2);
fprintf(1, 'n=%d, k=%d, Unsamped=%f\n', n, sum(sum(mask)),1-sum(sum(mask))/n/n); %
%imwrite(fftshift(mask), ['mask_77','.png']); 


% load lsq28; Img = imrotate(Img, -90); Img(:,end-6:end) = []; Img(:,1:7) = [];
% load lsq68;  Img = imrotate(Img, 90); Img(:,end-6:end) = []; Img(:,1:7) = [];
% load lsq200;  Img = imrotate(Img, 90); Img(:,end-6:end) = []; Img(:,1:7) = [];
% gt = 255*Img./max(abs(Img(:)));


gt = double(imread('images/lena.tif'));
% gt = double(imread('images/boats.tif'));
% gt = double(imread('images/cameraman.tif'));
% gt = double(imread('images/baboon.tif'));
% gt = double(imread('images/peppers.tif'));
% gt = double(imread('images/straw.tif'));

figure(334);imshow(abs(gt),[]);
% imwrite(uint8(abs(gt)), ['lsq28','.png']); 

% w = size(gt,2); w = w - mod(w, 2);`
% h = size(gt,1); h = h - mod(h, 2);
% gt = double(gt(1:h, 1:w, :)); % for some reason Caffe input needs even dimensions...
sigma_d = 0 * 255;
noise = randn(size(gt));
degraded = mask.*(fft2(gt) + noise * sigma_d + (0+1i)*noise * sigma_d); %
Im = ifft2(degraded); 
figure(22); imshow(Im,[],'Border','tight');
%figure(335);imshow(abs(Im),[]);

% load network for solver
% params.net = loadNet_qx3channel_diffSigma([size(gt),3], use_gpu);
 params.gt = gt;
% 
% params2.net = loadNet_qx3channel_diffSigma2([size(gt),3], use_gpu);
 params2.gt = gt;
 params3.gt = gt;

% run DAEP
params.sigma_net = 5;   %20;  %15;  %11;  %
params.num_iter = 600;
params2.sigma_net = 5; %25;   %20;  %15;  %11;  %
params2.num_iter = 600;
params3.sigma_net = 5;  %25;   %20;  %15;  %11;  %
params3.num_iter = 600;



load('E:\TDMSP_训练压缩感知\2019.4.9\data\MWCNN_GDSigma5_3_3Dgrey_400_input_threewavelet\MWCNN_GDSigma5_3_3Dgrey_400_input_threewavelet-epoch-44');  net1 = net;
net1 = dagnn.DagNN.loadobj(net1) ;
net1.removeLayer('objective') ;
out_idx = net1.getVarIndex('prediction') ;
net1.vars(net1.getVarIndex('prediction')).precious = 1 ;
net1.mode = 'test';
if gpu
    net1.move('gpu'); 
end


load('E:\TDMSP_训练压缩感知\2019.4.9\data\MWCNN_GDSigma5_3_3Dgrey_400_input_threewavelet\MWCNN_GDSigma5_3_3Dgrey_400_input_threewavelet-epoch-44');     net2 = net;
net2 = dagnn.DagNN.loadobj(net2) ;
net2.removeLayer('objective') ;
out_idx = net2.getVarIndex('prediction') ;
net2.vars(net2.getVarIndex('prediction')).precious = 1 ;
net2.mode = 'test';
if gpu
    net2.move('gpu');
end



load('E:\TDMSP_训练压缩感知\2019.4.9\data\MWCNN_GDSigma5_3_3Dgrey_400_input_threewavelet\MWCNN_GDSigma5_3_3Dgrey_400_input_threewavelet-epoch-44');    net3 = net;
net3 = dagnn.DagNN.loadobj(net3) ;
net3.removeLayer('objective') ;
out_idx = net3.getVarIndex('prediction') ;
net3.vars(net3.getVarIndex('prediction')).precious = 1 ;
net3.mode = 'test';
if gpu
    net3.move('gpu');
end





% map_deblur = Complex_DAEPMRIRec(Im, degraded, mask, sigma_d, params);
params.out_idx = out_idx;  params.gpu = gpu;  
params2.out_idx = out_idx;  params2.gpu = gpu;  
params3.out_idx = out_idx;  params3.gpu = gpu;  
[map_deblur,psnr_psnr] = Complex_DMSPMRIRec_natureimage_1sigma(Im, degraded, mask, sigma_d, params, params2, params3,net1, net2, net3);

[psnr4, ssim4, fsim4, ergas4, sam4] = MSIQA(abs(gt), abs(map_deblur));
[psnr4, ssim4, fsim4, ergas4, sam4]

figure(666);
subplot(2,3,[4,5,6]);imshow([abs(Im-gt)/255,abs(map_deblur-gt)/255],[]); title('Recon-error');colormap(jet);colorbar;
subplot(2,3,1);imshow(abs(gt)/255); title('Ground-truth');colormap(gray);
subplot(2,3,2);imshow(abs(Im)/255); title('Zero-filled');colormap(gray);
subplot(2,3,3);imshow(abs(map_deblur)/255); title('Net-recon');colormap(gray);
figure(667);imshow([real(gt)/255,imag(gt)/255,abs(gt)/255],[]); 
figure(668);imshow([abs(Im-gt)/255,abs(map_deblur-gt)/255],[]); colormap(jet);colorbar;
% imwrite(uint8(abs(map_deblur)), ['CartDAEPRec_qx3channel_15_25_015_lsq200','.png']); 

%  CartDAEPRec_qx3channel_15_25_015_lsq28 = map_deblur; save CartDAEPRec_qx3channel_15_25_015_lsq28; CartDAEPRec_qx3channel_15_25_015_lsq28;
%  CartDAEPRec_qx3channel_15_25_015_lsq68 = map_deblur; save CartDAEPRec_qx3channel_15_25_015_lsq68; CartDAEPRec_qx3channel_15_25_015_lsq68;
%  CartDAEPRec_qx3channel_15_25_015_lsq200 = map_deblur; save CartDAEPRec_qx3channel_15_25_015_lsq200; CartDAEPRec_qx3channel_15_25_015_lsq200;



