function dperf = forwardprop(dy,t,y,e,param)
%MSE.FORWARDPROP Forward propagate derivatives to performance
param.p = 10000;
% Copyright 2012-2015 The MathWorks, Inc.
if e<=0 % over-estimate (target < output)
    dperf = bsxfun(@times,dy,-2*e); % e = target-output
else % under-estimate (target < output)
    dperf = bsxfun(@times,dy,-2*e-param.p); % e = target-output
end
  
end
