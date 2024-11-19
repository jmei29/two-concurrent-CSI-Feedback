function [v, v_vec] = channel_varation_level_cal(H_a_d_pre, H_a_d, N_r)
v_vec = zeros(N_r, 1);
    for y = 1 : N_r
        H_a_d_y_pre = H_a_d_pre(:,:,y);
        H_a_d_y = H_a_d(:,:,y);
        v_vec(y) = norm(H_a_d_y_pre(:)'*H_a_d_y(:))/(norm(H_a_d_y_pre(:))*norm(H_a_d_y(:)));
    end % end of for y
    v = sum(v_vec)/N_r;
end % end of function