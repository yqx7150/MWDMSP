function Y = vl_nnloss(X,c,dzdy,varargin)

% --------------------------------------------------------------------
% Do the work x�Ǳ��� c��Ŀ��
% --------------------------------------------------------------------

if nargin <= 2 || isempty(dzdy)
    t = ((X-c).^2)/2;
    Y = sum(t(:)) ;
else
     Y = bsxfun(@minus,X, c).*dzdy; 
end