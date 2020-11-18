function perfs = apply(t,y,e,param)
%MSE.APPLY Calculate performances

% Copyright 2012-2015 The MathWorks, Inc.
% AH 2020/11: added condition to take into account price of lost demand
param.p = 10000;
if e<=0 % over-estimate (target < output)
    perfs = e .* e; % e = target-output
else % under-estimate (target < output)
    perfs = e .* e + param.p .* e; % e = target-output
end
    
end
