function [map,psnr_psnr] = CS_DMSP_multi_3_8_10(Im, degraded, mask, sigma_d, params, params2, params3, net1, net2 ,net3)
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



if ~any(strcmp('sigma_net',fieldnames(params)))
    params.sigma_net = 25; params2.sigma_net = 25;params3.sigma_net = 25;
end

if ~any(strcmp('num_iter',fieldnames(params)))
    params.num_iter = 400;params2.num_iter = 400;params3.num_iter = 400;
end

if ~any(strcmp('gamma',fieldnames(params)))
    params.gamma = 6.875; params2.gamma = 6.875;params3.gamma = 6.875;
end

if ~any(strcmp('mu',fieldnames(params)))
     params.mu = .9; params2.mu = .9;params3.mu = .9;
 %    params.mu = .09; params2.mu = .09;params3.mu = .09;
end

if ~any(strcmp('alpha',fieldnames(params)))
     params.alpha = .1; params2.alpha = .1;params3.alpha = .1;
  %   params.alpha = .01; params2.alpha = .01;params3.alpha = .01;
end


disp(params)

params.gamma = params.gamma * 4; params2.gamma = params2.gamma * 4;params3.gamma = params3.gamma * 4;

pad = [0, 0];
map = padarray(Im, pad, 'replicate', 'both');


% sigma_eta = sqrt(2) * params.sigma_net;
% relative_weight = params.gamma/(sigma_eta^2)/(params.gamma/(sigma_eta^2) + 1/(sigma_d^2));

step = zeros(size(map));



if any(strcmp('gt',fieldnames(params)))
    psnr = computePSNR(abs(params.gt), abs(map), pad);
    disp(['Initialized with PSNR: ' num2str(psnr)]);
end

for iter = 1:params.num_iter
    if any(strcmp('gt',fieldnames(params)))
        disp(['Running iteration: ' num2str(iter)]);
        tic();
    end
    
    prior_err_sum = zeros(size(map));
    repeat_num = 3;  %3;   %8;  %1; %12;
    for iiii = 1:repeat_num   
    % compute prior gradient 1
%     map_all = repmat(map,[1,1,3]);
%     input = real(map_all(:,:,[3,2,1])); % Switch channels for caffe    
%     noise = randn(size(input)) * params.sigma_net;
     map_real = real(map);
     input = prepare_date_3wavelet_padding(map_real/255,params.sigma_net);
%     rec = params.net.forward({input+noise});    
    rec = Processing_Im_w(single(input), net1, params.gpu, params.out_idx);
    rec = double(rec);
    prior_err = input - rec;   %{1};
    prior_err = prepare_date_1_3wavelet_padding(prior_err)*255;
    tmp = prior_err;    %wsy add
    prior_err = prior_err*0;%wsy add
    prior_err(pad(1)+1:end-pad(1), pad(2)+1:end-pad(2),:) = tmp(pad(1)+1:end-pad(1), pad(2)+1:end-pad(2),:);%wsy add
    %prior_err = prior_err(:,:,[3,2,1]);%wsy add
    prior_err1 = mean(prior_err,3);    %figure(2001);imshow(prior_err1,[])%wsy add
    
    
%     % compute prior gradient 2
%     input = imag(map_all(:,:,[3,2,1])); % Switch channels for caffe    
%     noise = randn(size(input)) * params.sigma_net;
%     %rec = params.net.forward({input+noise});    
    map_imag = imag(map);
    input = prepare_date_3wavelet_padding(map_imag/255,params.sigma_net);
    rec = Processing_Im_w(single(input), net1, params.gpu, params.out_idx);
    rec = double(rec);
    prior_err = input - rec;  %{1};     % figure(22);imshow([prior_err(:,:,1),prior_err(:,:,2),prior_err(:,:,3)],[])
    prior_err = prepare_date_1_3wavelet_padding(prior_err)*255;
    tmp = prior_err;    %wsy add
    prior_err = prior_err*0;%wsy add
    prior_err(pad(1)+1:end-pad(1), pad(2)+1:end-pad(2),:) = tmp(pad(1)+1:end-pad(1), pad(2)+1:end-pad(2),:);
    %prior_err = prior_err(:,:,[3,2,1]);
    prior_err2 = mean(prior_err,3);
    
    prior_err_sum = prior_err_sum + prior_err1+sqrt(-1)*prior_err2;    %figure(2001);imshow(abs(prior_err1+sqrt(-1)*prior_err2),[])
    end
    
       for iiii = 1:repeat_num   
    % compute prior gradient 1
%     map_all = repmat(map,[1,1,3]);
%     input = real(map_all(:,:,[3,2,1])); % Switch channels for caffe    
%     noise = randn(size(input)) * params.sigma_net;
     map_real = real(map);
     input = prepare_date_3wavelet_padding(map_real/255,params.sigma_net);
%     rec = params.net.forward({input+noise});    
    rec = Processing_Im_w(single(input), net1, params.gpu, params.out_idx);
    rec = double(rec);
    prior_err = input - rec;   %{1};
    prior_err = prepare_date_1_3wavelet_padding(prior_err)*255;
    tmp = prior_err;    %wsy add
    prior_err = prior_err*0;%wsy add
    prior_err(pad(1)+1:end-pad(1), pad(2)+1:end-pad(2),:) = tmp(pad(1)+1:end-pad(1), pad(2)+1:end-pad(2),:);%wsy add
    %prior_err = prior_err(:,:,[3,2,1]);%wsy add
    prior_err1 = mean(prior_err,3);    %figure(2001);imshow(prior_err1,[])%wsy add
    
    
%     % compute prior gradient 2
%     input = imag(map_all(:,:,[3,2,1])); % Switch channels for caffe    
%     noise = randn(size(input)) * params.sigma_net;
%     %rec = params.net.forward({input+noise});    
    map_imag = imag(map);
    input = prepare_date_3wavelet_padding(map_imag/255,params.sigma_net);
    rec = Processing_Im_w(single(input), net1, params.gpu, params.out_idx);
    rec = double(rec);
    prior_err = input - rec;  %{1};     % figure(22);imshow([prior_err(:,:,1),prior_err(:,:,2),prior_err(:,:,3)],[])
    prior_err = prepare_date_1_3wavelet_padding(prior_err)*255;
    tmp = prior_err;    %wsy add
    prior_err = prior_err*0;%wsy add
    prior_err(pad(1)+1:end-pad(1), pad(2)+1:end-pad(2),:) = tmp(pad(1)+1:end-pad(1), pad(2)+1:end-pad(2),:);
    %prior_err = prior_err(:,:,[3,2,1]);
    prior_err2 = mean(prior_err,3);
    
    prior_err_sum = prior_err_sum + prior_err1+sqrt(-1)*prior_err2;    %figure(2001);imshow(abs(prior_err1+sqrt(-1)*prior_err2),[])
    end
    
     for iiii = 1:repeat_num   
    % compute prior gradient 1
%     map_all = repmat(map,[1,1,3]);
%     input = real(map_all(:,:,[3,2,1])); % Switch channels for caffe    
%     noise = randn(size(input)) * params.sigma_net;
     map_real = real(map);
     input = prepare_date_3wavelet_padding(map_real/255,params.sigma_net);
%     rec = params.net.forward({input+noise});    
    rec = Processing_Im_w(single(input), net1, params.gpu, params.out_idx);
    rec = double(rec);
    prior_err = input - rec;   %{1};
    prior_err = prepare_date_1_3wavelet_padding(prior_err)*255;
    tmp = prior_err;    %wsy add
    prior_err = prior_err*0;%wsy add
    prior_err(pad(1)+1:end-pad(1), pad(2)+1:end-pad(2),:) = tmp(pad(1)+1:end-pad(1), pad(2)+1:end-pad(2),:);%wsy add
    %prior_err = prior_err(:,:,[3,2,1]);%wsy add
    prior_err1 = mean(prior_err,3);    %figure(2001);imshow(prior_err1,[])%wsy add
    
    
%     % compute prior gradient 2
%     input = imag(map_all(:,:,[3,2,1])); % Switch channels for caffe    
%     noise = randn(size(input)) * params.sigma_net;
%     %rec = params.net.forward({input+noise});    
    map_imag = imag(map);
    input = prepare_date_3wavelet_padding(map_imag/255,params.sigma_net);
    rec = Processing_Im_w(single(input), net1, params.gpu, params.out_idx);
    rec = double(rec);
    prior_err = input - rec;  %{1};     % figure(22);imshow([prior_err(:,:,1),prior_err(:,:,2),prior_err(:,:,3)],[])
    prior_err = prepare_date_1_3wavelet_padding(prior_err)*255;
    tmp = prior_err;    %wsy add
    prior_err = prior_err*0;%wsy add
    prior_err(pad(1)+1:end-pad(1), pad(2)+1:end-pad(2),:) = tmp(pad(1)+1:end-pad(1), pad(2)+1:end-pad(2),:);
    %prior_err = prior_err(:,:,[3,2,1]);
    prior_err2 = mean(prior_err,3);
    
    prior_err_sum = prior_err_sum + prior_err1+sqrt(-1)*prior_err2;    %figure(2001);imshow(abs(prior_err1+sqrt(-1)*prior_err2),[])
    end
    
    prior_err = prior_err_sum/repeat_num/3;
    
    % compute data gradient        %     map_conv = convn(map,rot90(kernel,2),'valid');
    data_err = zeros(size(prior_err));   %convn(map_conv-degraded,kernel,'full');

    % sum the gradients
    err = prior_err;  %relative_weight*prior_err + (1-relative_weight)*data_err;  %
   
    % update
    step = params.mu * step - params.alpha * err;
    map = map + step;
    
%     map_temp1 = medfilt2(real(map),[3,3]);
%     map_temp2 = medfilt2(imag(map),[3,3]);
%     map = map_temp1 + sqrt(-1)*map_temp2;
    
    temp_FFT = fft2(map);
    temp_FFT(mask==1) = degraded(mask==1);  %
    map = ifft2(temp_FFT);
    [psnr4, ssim4, ~] = MSIQA(abs(params.gt),  abs(map));
     psnr_psnr(iter,1)  = psnr4;
    
%     map = min(255,max(0,map));
%     if mod(iter,3)==0, figure(200+iter);imshow([abs(map)],[]);end
    
    if any(strcmp('gt',fieldnames(params)))
        psnr = computePSNR(abs(params.gt), abs(map), pad);
        disp(['PSNR is: ' num2str(psnr) ', iteration finished in ' num2str(toc()) ' seconds']);
    end
    
end
