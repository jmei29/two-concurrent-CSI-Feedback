function [phi_mat_quants, psi_mat_quants] = angle_quantization(phi_mat, psi_mat, sc_num, q_phi, q_psi)
%% reference paper
% Algorithm 1 in paper "DeepCSI: Rethinking Wi-Fi Radio Fingerprinting Through MU-MIMO CSI Feedback Deep Learning." arXiv preprint arXiv:2204.07614 (2022)
% based on equation (8)
% https://arxiv.org/abs/2204.07614
%% Input:
% q_phi: quantization bits for angle phi
% q_psi: quantization bits for angle psi
%% Inalization
num_psi_phi = size(phi_mat, 1);
phi_mat_quants = zeros(num_psi_phi, sc_num);
psi_mat_quants = zeros(num_psi_phi, sc_num);
for f = 1 : sc_num
    %% quantization of phi
    code_book_phi = 1/(2^q_phi) + (0 : 2^q_phi - 1)/(2^(q_phi - 1)) - 1;
    % in matlab function, -pi <= angle(*) <= pi, however, in the
    code_book_phi = code_book_phi * pi;
    partition_phi = code_book_phi(1 : end - 1);
    code_book_phi(2 : end) = code_book_phi(1 : end - 1);
    code_book_phi(1) = -pi;
    [index,phi_vec_quants] = quantiz(phi_mat(:, f)',partition_phi,code_book_phi);
    phi_mat_quants(:, f) = phi_vec_quants';
    %% quanztization of psi
    code_book_psi = 1/(2^(q_psi + 2)) + (0 : 2^q_psi - 1)/(2^(q_psi + 1));
    code_book_psi = code_book_psi * pi;
    partition_psi = code_book_psi(1 : end - 1);
    code_book_psi(2 : end) = code_book_psi(1 : end - 1);
    code_book_psi(1) = 0;
    [index,psi_vec_quants] = quantiz(psi_mat(:, f)', partition_psi, code_book_psi);
    psi_mat_quants(:, f) = psi_vec_quants';
end % end of for f
end % end of function