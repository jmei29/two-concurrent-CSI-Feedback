function [nmse_Phi, nmse_Psi] = accuracy_psi_phi_fb(phi_mat, psi_mat, phi_mat_fb, psi_mat_fb, sc_num)
%% input
% N_r x N_t x sc_num
% sc_num: number of total available subcarriers
%% calcualtion
nmse_Psi_vec = zeros(sc_num, 1);
nmse_Phi_vec = zeros(sc_num, 1);
for f = 1 : sc_num
    %% accuracy calculation
    nmse_Phi_vec(f) = norm(phi_mat(:,f) - phi_mat_fb(:,f))^2/norm(phi_mat(:,f))^2;
    if nmse_Phi_vec(f) > 1
        nmse_Phi_vec(f) = mean((abs(phi_mat(:,f) - phi_mat_fb(:,f)).^2)./...
            (max([abs(phi_mat(:,f)) abs(phi_mat_fb(:,f))],[],2).^2));
    end
    nmse_Psi_vec(f) = norm(psi_mat(:,f) - psi_mat_fb(:,f))^2/norm(psi_mat(:,f))^2;
    if nmse_Psi_vec(f) > 1
        nmse_Psi_vec(f) = mean((abs(psi_mat(:,f) - psi_mat_fb(:,f)).^2)./...
            (max([abs(psi_mat(:,f)) abs(psi_mat_fb(:,f))],[],2).^2));
    end
end % end of "for f = 1 : sc_num"
nmse_Phi = sum(nmse_Phi_vec)/sc_num;

nmse_Psi = sum(nmse_Psi_vec)/sc_num;
end % end of function