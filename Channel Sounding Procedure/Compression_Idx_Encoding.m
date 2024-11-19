function [Positional_Encoding_for_vec1, Positional_Encoding_for_vec2, Positional_Encoding_for_vec3] = ...
    Compression_Idx_Encoding(Compression_mode_idx, num_compression_modes)
%COMPRESSION_IDX_ENCODING Summary of this function goes here
%   Detailed explanation goes here
temp = Compression_mode_idx/num_compression_modes;
temp = temp * 0.5;
Positional_Encoding_for_vec1 = temp;
Positional_Encoding_for_vec2 = temp;
Positional_Encoding_for_vec3 = temp;
end

