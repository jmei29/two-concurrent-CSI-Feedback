function [util_value] = util_fun_nmse(nmse, nmse_low_bound_in_dB, nmse_up_bound_in_dB, TH_value)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
nmse_in_dB = 10*log10(nmse);
if nmse_in_dB >= nmse_up_bound_in_dB
    util_value = 0;
elseif nmse_in_dB > nmse_low_bound_in_dB && nmse_in_dB < nmse_up_bound_in_dB
    util_value = (nmse_up_bound_in_dB - nmse_in_dB)/(nmse_up_bound_in_dB - nmse_low_bound_in_dB);
    util_value = (1- TH_value) * util_value + TH_value;
else
    util_value = 1;
end
end

