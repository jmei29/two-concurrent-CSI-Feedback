function Memory_store(transition, str)
global SumTree_STA; % type: structure
global Memory_STA;
global SumTree_AP; % type: structure
global Memory_AP;

if strcmp(str, 'STA')
    max_p = max(SumTree_STA.tree(end - SumTree_STA.capacity + 1:end));
    if max_p == 0
        max_p = Memory_STA.abs_err_upper;
    end
    SumTree_add(max_p, transition, str)   % set the max p for new p
else
    % AP
    max_p = max(SumTree_AP.tree(end - SumTree_AP.capacity + 1:end));
    if max_p == 0
        max_p = Memory_AP.abs_err_upper;
    end
    SumTree_add(max_p, transition, str)   % set the max p for new p
end