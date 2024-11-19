function [util_value] = util_fun_CH_overhead(airtime_overhead, feedback_interval, MAX_value, MIN_Value)
%UTIL_FUN_CH_OVERHEAD Summary of this function goes here
%   Detailed explanation goes here
tau = airtime_overhead/feedback_interval;
if tau <= MIN_Value
    util_value = 1;
elseif tau > MIN_Value && tau < MAX_value
    util_value = (MAX_value - tau)/(MAX_value - MIN_Value);
else
    util_value = 0;
end
end

