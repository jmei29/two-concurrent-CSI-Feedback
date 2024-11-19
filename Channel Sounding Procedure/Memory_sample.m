function [b_idx, priorities, b_memory, IS_weights] = Memory_sample(n, str)
global SumTree_STA
global Memory_STA
global SumTree_AP
global Memory_AP

b_idx = zeros(n, 1);
b_memory = zeros(n, 1);
priorities = zeros(n, 1);
IS_weights = zeros(n, 1);
pri_seg = SumTree_total_p(str) / n;       % priority segment

% calculate beta
if strcmp(str, 'STA')
    Memory_STA.beta = min(1, Memory_STA.beta + ...
        Memory_STA.beta_increment_per_sampling);
    % min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight
    min_prob = min(SumTree_STA.tree(SumTree_STA.capacity : end,1)) / (pri_seg*n);
    beta = Memory_STA.beta;
else
    Memory_AP.beta = min(1, Memory_AP.beta + ...
        Memory_AP.beta_increment_per_sampling);
    min_prob = min(SumTree_AP.tree(SumTree_AP.capacity : end,1)) / (pri_seg*n);
    beta = Memory_AP.beta;
end
if min_prob == 0
    min_prob = 0.00001;
end
for i = 0 : n - 1
    a = pri_seg*i;
    b = pri_seg*(i+1);
    v = unifrnd(a,b);
    [idx, p, data] = SumTree_get_leaf(v, str);
    b_idx(i+1, 1) = idx;
    b_memory(i+1, 1) = data;
    priorities(i+1, 1) = p;
    prob = p / (pri_seg*n);
    IS_weights(i+1, 1) = (prob/min_prob)^(-beta);
end % end for
b_idx = b_idx(b_memory~=0);
priorities = priorities(b_memory~=0);
IS_weights = IS_weights(b_memory~=0);
b_memory = b_memory(b_memory~=0);
end