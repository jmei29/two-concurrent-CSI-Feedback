function [phi_mat, psi_mat, idx_phi, idx_psi, num_psi_phi] = ...
    spatial_compression(h_CFR_3D_mat, sc_num, N_t, N_r)
%% input
% h_CFR_3D_mat: the the 3D matrix CFR between AP and STA, where size is
% N_r x N_t x sc_num
% sc_num: number of total available subcarriers
% N_t: number of transmitting antennas at the AP side
% N_r: number of receving antennas at the STA side
%% reference paper
% Algorithm 1 in paper "DeepCSI: Rethinking Wi-Fi Radio Fingerprinting Through MU-MIMO CSI Feedback Deep Learning." 
% arXiv preprint arXiv:2204.07614 (2022)
%% Inalization
num_psi_phi = (2*N_t*N_r - (N_r)^2 - N_r)/2;
psi_mat = zeros(num_psi_phi, sc_num);
phi_mat = zeros(num_psi_phi, sc_num);
%% calcualtion
for f = 1 : sc_num
    idx_psi = 1;
    idx_phi = 1;
    h_CFR_mat = h_CFR_3D_mat(:, :, f); % N_r x N_t
    % SVD decomposition of CFR matrix
    [U,S,V_] = svd(h_CFR_mat);
    % generate V matrix
    V_f = V_(:, 1 : N_r);% N_t x N_r
    % generate matrix D_k
    v_temp = V_f(N_t, 1 : N_r);
    for ii = 1 : N_r
        v_temp(ii) = exp(1i*angle(v_temp(ii)));
    end % end for ii
    D_f_0 = diag(v_temp); % D_f_0 is not sent back to the AP side.
    % Generate matrix Omega
    Omega_f = V_f * D_f_0'; % N_t x N_r
    for i = 1 : min(N_r, N_t - 1)
        %% generate value for phi
        D_f_i = diag(ones(1, N_t));
        for l = i : N_t - 1
            phi_mat(idx_phi, f) = angle(Omega_f(l, i));
            % in matlab function, -pi <= angle(*) <= pi, however, in the
            % standard, 0 <= phi <= 2*pi
            D_f_i(l, l) = exp(1i*phi_mat(idx_phi, f));
            idx_phi = idx_phi + 1;
        end % end of "for l = i : N_t - 1"
        % Update Omega matrix
        Omega_f = D_f_i' * Omega_f;
        %% generate value for psi
        for l = i+1 : N_t
            psi_mat(idx_psi, f) = acos(abs(Omega_f(i, i))/sqrt((abs(Omega_f(i, i)))^2 + (abs(Omega_f(l, i)))^2));
            if imag(psi_mat(idx_psi, f)) ~= 0
                fprintf('error\n')
            end
            % generate matrix G_f_l_i
            G_f_l_i = diag(ones(1, N_t));
            G_f_l_i(i, i) = cos(psi_mat(idx_psi, f));
            G_f_l_i(l, l) = cos(psi_mat(idx_psi, f));
            G_f_l_i(l, i) = -sin(psi_mat(idx_psi, f));
            G_f_l_i(i, l) = sin(psi_mat(idx_psi, f));
            % update Omega matrix
            Omega_f = G_f_l_i * Omega_f;
            % update index
            idx_psi = idx_psi + 1;
        end % end of "for l = i+1 : N_t"
    end % end of for i = 1 : min(N_r, N_t - 1)
end % end of for f = 1 : sc_num
% RESET idx_phi and idx_psi
idx_phi = idx_phi - 1;
idx_psi = idx_psi - 1;
end % end of function