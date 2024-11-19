function SumTree_update(tree_idx, p, str)
global SumTree_AP; % type: structure
global SumTree_STA; % type: structure

if strcmp(str, 'STA')
    change = p - SumTree_STA.tree(tree_idx, 1);
    SumTree_STA.tree(tree_idx, 1) = p;
    % then propagate the change through tree
    tree_idx_ = tree_idx - 1;
    while tree_idx_ ~= 0    % this method is faster than the recursive loop in the reference code
        tree_idx_ = floor((tree_idx_ - 1)/2);
        tree_idx = tree_idx_ + 1;
        SumTree_STA.tree(tree_idx,1) = SumTree_STA.tree(tree_idx,1) + change;
    end
else
    change = p - SumTree_AP.tree(tree_idx, 1);
    SumTree_AP.tree(tree_idx, 1) = p;
    % then propagate the change through tree
    tree_idx_ = tree_idx - 1;
    while tree_idx_ ~= 0    % this method is faster than the recursive loop in the reference code
        tree_idx_ = floor((tree_idx_ - 1)/2);
        tree_idx = tree_idx_ + 1;
        SumTree_AP.tree(tree_idx,1) = SumTree_AP.tree(tree_idx,1) + change;
    end
end
end