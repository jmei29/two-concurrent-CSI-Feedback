function [phi_mat_fb, psi_mat_fb] = tone_grouping(phi_mat_quants, psi_mat_quants, sc_num, N_g)
%% inialitzation
num_psi_phi = size(phi_mat_quants, 1);
phi_mat_fb = zeros(num_psi_phi, sc_num);
psi_mat_fb = zeros(num_psi_phi, sc_num);
%% processing
compress_sc_num = floor(sc_num/N_g);
for i = 1 : compress_sc_num
    idx = min(max(floor((i-0.5)*N_g), 1), sc_num);
    phi_mat_fb(:, (i-1)*N_g + 1: i*N_g) = repmat(phi_mat_quants(:, idx), 1, N_g);
    psi_mat_fb(:, (i-1)*N_g + 1: i*N_g) = repmat(psi_mat_quants(:, idx), 1, N_g);
end % end of "for i = 1 : N_g"
end % end of function