function [map,psnr_ssim,psnr_psnr] = TDMSP_deblur(degraded, kernel, sigma_d, params, net1,aa)
% Implements stochastic gradient descent (SGD) maximum-a-posteriori for image deblurring described in:
% S. A. Bigdeli, M. Zwicker, "Image Restoration using Autoencoding Priors".
%
%
% Input:
% degraded: Observed degraded RGB input image in range of [0, 255].
% kernel: Blur kernel (internally flipped for convolution).
% sigma_d: Noise standard deviation.
% params: Set of parameters.
% params.net: The DAE Network object loaded from MatCaffe.
%
% Optional parameters:
% params.sigma_net: The standard deviation of the network training noise. default: 25
% params.num_iter: Specifies number of iterations.
% params.gamma: Indicates the relative weight between the data term and the prior. default: 6.875
% params.mu: The momentum for SGD optimization. default: 0.9
% params.alpha the step length in SGD optimization. default: 0.1
%
%
% Outputs:
% map: Solution.


% if ~any(strcmp('net',fieldnames(params)))
%     error('Need a DAE network in params.net!');
% end
if ~any(strcmp('sigma_dae',fieldnames(params)))
    params.sigma_dae = 11;
end

if ~any(strcmp('num_iter',fieldnames(params)))
    params.num_iter = 300;
end

if ~any(strcmp('mu',fieldnames(params)))
    params.mu = .9;
end

if ~any(strcmp('alpha',fieldnames(params)))
    params.alpha = .1;
end


disp(params)

pad = floor(size(kernel)/2);
map = padarray(degraded, pad, 'replicate', 'both');

step = zeros(size(map));

if any(strcmp('gt',fieldnames(params)))
    map_center = map(pad(1)+1:end-pad(1),pad(2)+1:end-pad(2),:);
%     psnr = computePSNR(params.gt, map_center, pad);
%     disp(['Initialized with PSNR: ' num2str(psnr)]);
end

psnr_ssim = zeros(params.num_iter,1);
for iter = 1:params.num_iter
    if any(strcmp('gt',fieldnames(params)))
        disp(['Running iteration: ' num2str(iter)]);
        tic();
    end
    
    prior_err_sum = zeros(size(map));
    repeat_num = 3;  %3;   %8;  %1; %12;
    
    for iiii = 1:repeat_num
    % compute prior gradient
    map_all = repmat(map,[1,1,1]);
    input = prepare_date_3wavelet_padding(map_all/255,params.sigma_net);  
    rec = Processing_Im_w(single(input), net1, params.gpu, params.out_idx);
    rec = double(rec);
    prior_err = input - rec;   %{1};
    prior_err = prepare_date_1_3wavelet_padding(prior_err)*255;
    tmp = prior_err;    %wsy add
    prior_err = prior_err*0;%wsy add
    prior_err(pad(1)+1:end-pad(1), pad(2)+1:end-pad(2),:) = tmp(pad(1)+1:end-pad(1), pad(2)+1:end-pad(2),:);%wsy add
    prior_err = mean(prior_err,3); 

    prior_err_sum = prior_err_sum + prior_err;
     
    end
    for iiii = 1:repeat_num
    % compute prior gradient
    map_all = repmat(map,[1,1,1]);
    input = prepare_date_3wavelet_padding(map_all/255,params.sigma_net);  
    rec = Processing_Im_w(single(input), net1, params.gpu, params.out_idx);
    rec = double(rec);
    prior_err = input - rec;   %{1};
    prior_err = prepare_date_1_3wavelet_padding(prior_err)*255;
    tmp = prior_err;    %wsy add
    prior_err = prior_err*0;%wsy add
    prior_err(pad(1)+1:end-pad(1), pad(2)+1:end-pad(2),:) = tmp(pad(1)+1:end-pad(1), pad(2)+1:end-pad(2),:);%wsy add
    prior_err = mean(prior_err,3); 
    prior_err_sum = prior_err_sum + prior_err;
     
    end
    for iiii = 1:repeat_num
    % compute prior gradient
    map_all = repmat(map,[1,1,1]);
    input = prepare_date_3wavelet_padding(map_all/255,params.sigma_net);  
    rec = Processing_Im_w(single(input), net1, params.gpu, params.out_idx);
    rec = double(rec);
    prior_err = input - rec;   %{1};
    prior_err = prepare_date_1_3wavelet_padding(prior_err)*255;
    tmp = prior_err;    %wsy add
    prior_err = prior_err*0;%wsy add
    prior_err(pad(1)+1:end-pad(1), pad(2)+1:end-pad(2),:) = tmp(pad(1)+1:end-pad(1), pad(2)+1:end-pad(2),:);%wsy add
    prior_err = mean(prior_err,3); 
    prior_err_sum = prior_err_sum + prior_err;   
    end

    prior_err = prior_err_sum/repeat_num/3/3;
%     figure(333);imshow(prior_err,[])
    % compute data gradient
    map_conv = convn(map,rot90(kernel,2),'valid');
    data_err = map_conv-degraded;
    data_grad = convn(data_err,kernel,'full');
    if sigma_d<0
        sigma2 = 2*params.sigma_dae*params.sigma_dae;
        lambda = (numel(degraded))/(sum(data_err(:).^2) + numel(degraded)*sigma2*sum(kernel(:).^2));
        relative_weight = (lambda)/(lambda + 1/params.sigma_dae/params.sigma_dae);
    else
        relative_weight = (1/sigma_d/sigma_d)/(1/sigma_d/sigma_d + 1/params.sigma_dae/params.sigma_dae);
    end
    % sum the gradients
    grad_joint = data_grad*relative_weight + prior_err*(1-relative_weight);
   
    % update
    step = params.mu * step - params.alpha * grad_joint;
    map = map + step;
    map = min(255,max(0,map));
%     figure(444); imshow(map,[]);
%     if mod(iter,20)==0, figure(200+iter);imshow([uint8(map_center)],[]);end

    [psnr4, ssim4, ~] = MSIQA(params.gt,  map(pad(1)+1:end-pad(1),pad(2)+1:end-pad(2),:));
%     psnr_ssim(iter,1)  = iter;
    psnr_psnr(iter,1)  = psnr4;
    psnr_ssim(iter,1)  = ssim4;
    
    
    if any(strcmp('gt',fieldnames(params)))
        map_center = map(pad(1)+1:end-pad(1),pad(2)+1:end-pad(2),:);
        psnr = csnr(params.gt, map_center, 0,0);
%         ssim=cal_ssim(params.gt/255, map_center/255, 0,0);
         ssim=cal_ssim(params.gt, map_center, 0,0);
        disp(['PSNR is: ' num2str(psnr)  'SSIM is: ' num2str(ssim)   ', iteration finished in ' num2str(toc()) ' seconds']);
    end
 end
end