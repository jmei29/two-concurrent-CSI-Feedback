function [V_recon_3D_mat] = reconstruction(phi_mat_fb, psi_mat_fb, sc_num, N_t, N_r)
%% reference paper
% Algorithm 1 in paper "DeepCSI: Rethinking Wi-Fi Radio Fingerprinting Through MU-MIMO CSI Feedback Deep Learning." arXiv preprint arXiv:2204.07614 (2022)
% based on eqaution (7)
% https://arxiv.org/abs/2204.07614
%% Input:
% phi_mat_fb: matrix of angle phi in the feedback report
% psi_mat_fb: matrix of angle psi in the feedback report
% sc_num: total number of subcarriers/tones
% N_t: number of transmit antennas at the AP side
% N_r: number of receiving antennas at the STA side
%% Operation
V_recon_3D_mat = zeros(N_t, N_r, sc_num);
for f = 1 : sc_num
    idx_psi = 1;
    idx_phi = 1;
    %% initialize beamforming feedback matrix V_f
    for i = 1 : min(N_r, N_t - 1)
        %% generate matrix D_f_i
        D_f_i = diag(ones(1, N_t));
        for l = i : N_t - 1
            D_f_i(l, l) = exp(1i*phi_mat_fb(idx_phi, f));
            idx_phi = idx_phi + 1;
        end % % end of "for l = i : N_t - 1"
        temp_mat = D_f_i;
        %% generate matrix G_f_l_i
        for l = i+1 : N_t
            % generate matrix G_f_l_i
            G_f_l_i = diag(ones(1, N_t));
            G_f_l_i(i, i) = cos(psi_mat_fb(idx_psi, f));
            G_f_l_i(l, l) = cos(psi_mat_fb(idx_psi, f));
            G_f_l_i(l, i) = -sin(psi_mat_fb(idx_psi, f));
            G_f_l_i(i, l) = sin(psi_mat_fb(idx_psi, f));
            % update index
            idx_psi = idx_psi + 1;
            % update template matrix
            temp_mat = temp_mat * G_f_l_i.';
        end % end of for l = i+1 : N_t
        %% update beaforming feedback matrix
        if i == 1
            V_recon_mat = temp_mat;
        else
            V_recon_mat = V_recon_mat * temp_mat;
        end
    end % end of loop i
    V_recon_mat = V_recon_mat * eye(N_t, N_r);
    V_recon_3D_mat(:,:,f) = V_recon_mat;
end % end of loop for f
end % end of function