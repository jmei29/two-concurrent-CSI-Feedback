function [total_p] = SumTree_total_p(str)
global SumTree_STA; % type: structure
global SumTree_AP; % type: structure

if strcmp(str, 'STA')
    total_p = SumTree_STA.tree(1, 1);
else
    total_p = SumTree_AP.tree(1, 1);
end
end