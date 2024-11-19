function [leaf_idx, p, data] = SumTree_get_leaf(v, str)
global SumTree_STA; % type: structure
global SumTree_AP; % type: structur
if strcmp(str, 'STA')
    parent_idx = 0;
    while 1     % the while loop is faster than the method in the reference code
        cl_idx = 2 * parent_idx + 1 ;        % this leaf's left and right kids
        cr_idx = cl_idx + 1;
        if cl_idx >= 2*SumTree_STA.capacity - 1     % reach bottom, end search
            leaf_idx = parent_idx;
            break
        else    % downward search, always search for a higher priority node
            if v <= SumTree_STA.tree(cl_idx + 1, 1)
                parent_idx = cl_idx;
            else
                v = v - SumTree_STA.tree(cl_idx + 1, 1);
                parent_idx = cr_idx;
            end
        end
    end

    leaf_idx = leaf_idx + 1;
    data_idx = leaf_idx - SumTree_STA.capacity + 1;
    p = SumTree_STA.tree(leaf_idx, 1);
    data = SumTree_STA.data(data_idx, 1);
else
    parent_idx = 0;
    while 1     % the while loop is faster than the method in the reference code
        cl_idx = 2 * parent_idx + 1 ;        % this leaf's left and right kids
        cr_idx = cl_idx + 1;
        if cl_idx >= 2*SumTree_AP.capacity - 1     % reach bottom, end search
            leaf_idx = parent_idx;
            break
        else    % downward search, always search for a higher priority node
            if v <= SumTree_AP.tree(cl_idx + 1, 1)
                parent_idx = cl_idx;
            else
                v = v - SumTree_AP.tree(cl_idx + 1, 1);
                parent_idx = cr_idx;
            end
        end
    end

    leaf_idx = leaf_idx + 1;
    data_idx = leaf_idx - SumTree_AP.capacity + 1;
    p = SumTree_AP.tree(leaf_idx, 1);
    data = SumTree_AP.data(data_idx, 1);
end
end