close all;
clear;
clc;

addpath(genpath('E:\TDAEP_MWCNN多通道去马赛克\DAEP_MWCNN1'));

gpu = 1;

%% Deblurring demo
% load image and kernel
load('kernels.mat');


for ImgNo =1 : 6% 1:9
        switch ImgNo
            case 1
                fn1 = 'lena';
                fn =double(imread('08.png'));
            case 2
                fn1 = 'boats';
                fn = double(imread('boats.tif'));
            case 3
                fn1 = 'cameraman';
                fn = double(imread('cameraman.tif'));
            case 4
                fn1 ='baboon';
                fn = double(imread('baboon.tif'));
            case 5
                fn1 ='peppers';
                fn = double(imread('peppers.tif'));
            case 6
                fn1 ='straw';
                fn = double(imread('straw.tif'));
        end
gt = fn;
w = size(gt,2); w = w - mod(w, 2);
h = size(gt,1); h = h - mod(h, 2);
gt = double(gt(1:h, 1:w, :)); % for some reason Caffe input needs even dimensions...
%gt是原图补了0的
for kernel_a = 1 : 2
      switch kernel_a
         case 1
              aa = 1717;
              kernel = kernels{2};
         case 2
              aa =2525;
              kernel = kernels{8};
      end
for alpha =  1 :2
    switch alpha
            case 1
                a1 = 1;
                a = 0.01;
            case 2
                a1 = 3;
                a = 0.03;
    end
sigma_d = 255 * a;

pad = floor(size(kernel)/2);   %% 这里是我改的 floor(size(kernel)/2);
gt_extend = padarray(gt, pad, 'replicate', 'both');

degraded =convn(gt_extend, rot90(kernel,2), 'valid') ;

noise = randn(size(degraded));
degraded = degraded + noise * sigma_d;
figure(11);imshow(gt,[])
figure(22); imshow(degraded,[],'Border','tight');


for ii = 1 : 4
    switch ii
        case 1
            sigma_dae = 1;
        case 2
            sigma_dae = 2;
        case 3
            sigma_dae = 3;
        case 4
            sigma_dae = 4;
    end
    
params.sigma_dae =sigma_dae;
params.gt = gt;
params.sigma_net = 5;   %20;  %15;  %11;  %
sigma = params.sigma_net;
params.num_iter = 500;
for i = 45 
% load('E:\多通道测试_data\MWCNN_GDSigma15_3_3Dgrey_400\MWCNN_GDSigma15_3_3Dgrey_400-epoch-1');  net1 = net;
mymodel = (['E:\TDAEP_MWCNN多通道去马赛克\MWCNN_GDSigma5_3_3Dgrey_400_input_threewavelet\MWCNN_GDSigma5_3_3Dgrey_400_input_threewavelet-epoch-'  num2str(i,'%d') ]); 
load(mymodel);
net1 = net;
net1 = dagnn.DagNN.loadobj(net1) ;
net1.removeLayer('objective') ;
out_idx = net1.getVarIndex('prediction') ;
net1.vars(net1.getVarIndex('prediction')).precious = 1 ;
net1.mode = 'test';
if gpu
    net1.move('gpu'); 
end

params.out_idx = out_idx;  params.gpu = gpu;
         
 
%% 现在的DAEP
[map_deblur_extend,psnr_ssim,psnr_psnr ]= TDMSP_deblur(degraded, kernel, sigma_d, params, net1 ,aa);
% map_deblur_extend = DAEP_deblurmulti_mwcnn(degraded, kernel, sigma_d, params, params2,net1, net2,net1_backword,net2_backword);
map_deblur = map_deblur_extend(pad(1)+1:end-pad(1),pad(2)+1:end-pad(2),:);
[psnr4, ssim4, fsim4, ergas4, sam4] = MSIQA(gt, map_deblur);
[psnr4, ssim4, fsim4, ergas4, sam4]

%  save (['./debulur_sigma8/',fn1,'psnr_ssim_' num2str(i,'%d') ,'_simga',num2str(sigma,'%d'),'kernels',num2str(aa,'%d'),'_alpha',num2str(a1,'%d')],'psnr_ssim');
%  save (['./debulur_sigma8/',fn1,'psnr_psnr_' num2str(i,'%d') ,'_simga',num2str(sigma,'%d'),'kernels',num2str(aa,'%d'),'-alpha',num2str(a1,'%d')],'psnr_psnr');
 save (['./debulur_sigma5/',fn1,'_simga',num2str(sigma,'%d'),'_simgadac',num2str(sigma_dae,'%d'),'_epoch',num2str(i,'%d'),'kernels',num2str(aa,'%d'),'_alpha',num2str(a1,'%d')]);
 
% figure;
% subplot(131);
% imshow(gt/255); title('Ground Truth')
% subplot(132);
 %imshow(degraded/255); title('Blurry')
 
 %subplot(133);
% imshow(map_deblur/255); title('Restored')

end
end
end
end
end