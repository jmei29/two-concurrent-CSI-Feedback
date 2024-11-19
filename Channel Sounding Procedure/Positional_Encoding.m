function [Positional_Encoding_vec1, Positional_Encoding_vec2, Positional_Encoding_vec3] = ...
    Positional_Encoding(Compression_mode_idx, N_c, AP_antenna_num, STA_antenna_num)
Positional_Encoding_vec = zeros(N_c + AP_antenna_num + STA_antenna_num, 1);
for i = 1 : N_c + AP_antenna_num + STA_antenna_num
    if mod(i,2) == 0
        w = 1/(10000^(i/(N_c + AP_antenna_num + STA_antenna_num)));
        Positional_Encoding_vec(i,1) = sin(w*Compression_mode_idx);
    else
        w = 1/(10000^((i - 1)/(N_c + AP_antenna_num + STA_antenna_num)));
        Positional_Encoding_vec(i,1) = cos(w*Compression_mode_idx);
    end
end % end of for
Positional_Encoding_vec1 = Positional_Encoding_vec(1 : N_c,:);
Positional_Encoding_vec2 = Positional_Encoding_vec(N_c + 1 : N_c + AP_antenna_num,:);
Positional_Encoding_vec3 = Positional_Encoding_vec(N_c + AP_antenna_num + 1 : end,:);
end % end of function