function [nmse] = accuracy_CSI_fb(V_recon_3D_mat, h_CFR_3D_mat, sc_num, N_t, N_r, mode)
%% input
% h_CFR_3D_mat: the the 3D matrix CFR between AP and STA, where size is
% N_r x N_t x sc_num
% sc_num: number of total available subcarriers
% N_t: number of transmitting antennas at the AP side
% N_r: number of receving antennas at the STA side
% mode: if mode=0, CFR CSI feedback; if mode=1, time domain CSI feedback.
%% calcualtion
nmse_per_SC = zeros(sc_num, 1);
if mode == 0
    %% CSI feedback in the spatial-frequency domain
    for f = 1 : sc_num
        %% SVD decomposition of CFR matrix
        h_CFR_mat = h_CFR_3D_mat(:, :, f); % N_r x N_t
        V_recon_mat = V_recon_3D_mat(:, :, f); % N_t x N_r
        [U,S,V_] = svd(h_CFR_mat);
        % generate V matrix
        V_f = V_(:, 1 : N_r);% N_t x N_r
        % generate matrix D_k
        v_temp = V_f(N_t, 1 : N_r);
        for ii = 1 : N_r
            v_temp(ii) = exp(1i*angle(v_temp(ii)));
        end % end for ii
        D_f_0 = diag(v_temp); % D_f_0 is not sent back to the AP side.
        V_recon_mat = V_recon_mat * D_f_0;
        %% accuracy calculation
        nmse_per_SC(f) = norm(V_recon_mat - V_f, 2)^2/norm(V_recon_mat, 2)^2;
        %% if nmse_per_SC is larger than 1
        if nmse_per_SC(f) > 1
            nmse_per_SC(f) = mean((abs(V_recon_mat(:) - V_f(:)).^2)./(max([abs(V_recon_mat(:)) abs(V_f(:))],[],2).^2));
        end
    end % end of "for f = 1 : sc_num"
else
    %% CSI feedback in the time domain
    for f = 1 : sc_num
        h_CFR_mat = h_CFR_3D_mat(:, :, f); % N_r x N_t
        V_recon_mat = V_recon_3D_mat(:, :, f); % N_t x N_r
        [U,S,V_] = svd(h_CFR_mat);
        % generate V matrix
        V_f = V_(:, 1 : N_r);% N_t x N_r
        %% accuracy calculation
        nmse_per_SC(f) = norm(V_recon_mat - V_f, 2)^2/norm(V_recon_mat, 2)^2;
    end
end % end if mode
nmse = sum(nmse_per_SC)/sc_num;
end % end of function