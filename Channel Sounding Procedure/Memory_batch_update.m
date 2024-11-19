function Memory_batch_update(tree_idx, abs_errors, str)
global Memory_STA
global Memory_AP

if strcmp(str, 'STA')
    abs_errors = abs_errors + Memory_STA.epsilon;  % convert to abs and avoid 0
    clipped_errors = min(abs_errors, Memory_STA.abs_err_upper);
    ps = clipped_errors.^(Memory_STA.alpha);
    num_batch = length(tree_idx);
    for i = 1 : num_batch
        SumTree_update(tree_idx(i), ps(i), str);
    end % for i = 1 : num_batch
else
    
    abs_errors = abs_errors + Memory_AP.epsilon;  % convert to abs and avoid 0
    clipped_errors = min(abs_errors, Memory_AP.abs_err_upper);
    ps = clipped_errors.^(Memory_AP.alpha);
    num_batch = length(tree_idx);
    for i = 1 : num_batch
        SumTree_update(tree_idx(i), ps(i), str);
    end % for i = 1 : num_batch
end
end