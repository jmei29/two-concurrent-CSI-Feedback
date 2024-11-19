function SumTree_add(p, CH_data_idx, str)
global SumTree_AP; % type: structure
global SumTree_STA; % type: structure

if strcmp(str, 'STA')
    tree_idx = SumTree_STA.data_pointer + SumTree_STA.capacity - 1;
    SumTree_STA.data(SumTree_STA.data_pointer, 1) = CH_data_idx;  % update data_frame
    SumTree_update(tree_idx, p, str)  % update tree_frame
    
    SumTree_STA.data_pointer = SumTree_STA.data_pointer + 1;
    if SumTree_STA.data_pointer > SumTree_STA.capacity
        % replace when exceed the capacity
        SumTree_STA.data_pointer = 1;
    end
else
    tree_idx = SumTree_AP.data_pointer + SumTree_AP.capacity - 1;
    SumTree_AP.data(SumTree_AP.data_pointer, 1) = CH_data_idx;  % update data_frame
    SumTree_update(tree_idx, p, str)  % update tree_frame
    
    SumTree_AP.data_pointer = SumTree_AP.data_pointer + 1;
    if SumTree_AP.data_pointer > SumTree_AP.capacity
        % replace when exceed the capacity
        SumTree_AP.data_pointer = 1;
    end
end