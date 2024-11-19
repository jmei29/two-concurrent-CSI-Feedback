%% Description:
%  Created by Jie Mei, Sept. 12, 2022
%  Revised by Jie Mei, May 13
%%
clear; clc;
addpath('../CSI data generation/CSI-data-set-generation')% load function path of cost2100 channel model
%% Scenario Parametersl
CASE_IDX = 4;
openfile = sprintf('../simualtion_parameters_CASE_%d.mat', CASE_IDX);
load(openfile, 'parameters');
%% Training Settings
training_CASE = 145;
% CASE 2" every action is random
% CASE 3: new reward function
%% Parameters
cache_length_from_matlab_to_python = 64;
file_name = sprintf('/DQN/Training_Case_ID_%d', training_CASE);
file_save_dir = strcat(parameters.result_data_saving_path, file_name);
if ~exist(file_save_dir, 'dir')
    mkdir(file_save_dir)
end
DNN_file_save_dir = strcat(parameters.DNN_saving_path, file_name);
if ~exist(DNN_file_save_dir, 'dir')
    mkdir(DNN_file_save_dir)
end
%% paramerters defintion for CSI feedback in the main
CH_sounding_idx = 1;
CFI_fedback_idx = 1;
learning_step = 1;
snap_idx = 1;
total_simulated_CH_num = 10000;
learning_Step_Max = total_simulated_CH_num-1000;
%% Validation Section
if_perform_validate = 1;
collect_data_num = 900;
return_val_index = 1;
validation_sample_num = 0;
return_val_discount_record = zeros(collect_data_num, 1);
Q_est_from_AP_side_versus_return = zeros(total_simulated_CH_num, 1);
Q_est_from_STA_side_versus_return = zeros(total_simulated_CH_num, 1);
discount_factor = 0.6;
trajectory_len = 9;
%% PER
%---------------STA side----------------------
global SumTree_STA
SumTree_STA.data_pointer = 1;
SumTree_STA.capacity = total_simulated_CH_num;
SumTree_STA.tree = zeros(2*SumTree_STA.capacity - 1, 1);
SumTree_STA.data = zeros(SumTree_STA.capacity, 1);  % stor indexes of data
global Memory_STA
Memory_STA.epsilon = 0.01;  % small amount to avoid zero priority
Memory_STA.alpha = 0.6;  % [0~1] convert the importance of TD error to priority
Memory_STA.beta = 0.4;  % importance-sampling, from initial value increasing to 1
Memory_STA.beta_increment_per_sampling = 0.001;
Memory_STA.abs_err_upper = 1.0;  % clipped abs error
STA_Sample_p_weights_record = - ones(total_simulated_CH_num, cache_length_from_matlab_to_python);
%---------------AP side----------------------
global SumTree_AP
SumTree_AP.data_pointer = 1;
SumTree_AP.capacity = total_simulated_CH_num;
SumTree_AP.tree = zeros(2*SumTree_AP.capacity - 1, 1);
SumTree_AP.data = zeros(SumTree_AP.capacity, 1);  % stor indexes of data
global Memory_AP
Memory_AP.epsilon = 0.01;  % small amount to avoid zero priority
Memory_AP.alpha = 0.6;  % [0~1] convert the importance of TD error to priority
Memory_AP.beta = 0.4;  % importance-sampling, from initial value increasing to 1
Memory_AP.beta_increment_per_sampling = 0.001;
Memory_AP.abs_err_upper = 1.0;  % clipped abs error
AP_Sample_p_weights_record = - ones(total_simulated_CH_num, cache_length_from_matlab_to_python);
%% Variables
CF = cell(3, 1); % channel features
CFI = cell(3, 1);
duration_of_two_CFI_fed = 0;
cumulative_reward = 0;
if_train_DRL = 0;
% STA's Record
CF_record = zeros(total_simulated_CH_num, parameters.N_c + parameters.AP_antenna_num + ...
    parameters.STA_antenna_num);
CF_real_record = zeros(total_simulated_CH_num, parameters.N_c + parameters.AP_antenna_num + ...
    parameters.STA_antenna_num);
action_record_STA_side = zeros(total_simulated_CH_num, 1);
reward_fun_record_STA_side = zeros(total_simulated_CH_num, 1);
reward_fun_time_average_record_STA_side = zeros(total_simulated_CH_num, 1);
LOSS_record_STA_side = zeros(learning_Step_Max, 1);
learning_rate_record_STA_side = zeros(learning_Step_Max, 1);
utility_value_per_sounding_record = zeros(total_simulated_CH_num, 1);
CH_airtime_overhead = zeros(total_simulated_CH_num, 1);
CH_overhead = zeros(total_simulated_CH_num, 1);
CH_overhead_util = zeros(total_simulated_CH_num, 4);
CFI_overhead = zeros(total_simulated_CH_num, 1);
nmse_record = zeros(total_simulated_CH_num, 1);
nmse_util_record = zeros(total_simulated_CH_num, 3);
state_action_values_STA_record = single(zeros(total_simulated_CH_num, 8));
% AP's Record
CFI_record = zeros(total_simulated_CH_num, parameters.N_c + parameters.AP_antenna_num + ...
    parameters.STA_antenna_num);
action_record_AP_side = zeros(total_simulated_CH_num, 1);
reward_fun_record_AP_side = zeros(total_simulated_CH_num, 1);
reward_fun_time_average_record_AP_side = zeros(total_simulated_CH_num, 1);
LOSS_record_AP_side = zeros(learning_Step_Max, 1);
state_action_values_AP_record = single(zeros(total_simulated_CH_num, parameters.num_of_CSI_compression_modes));
learning_rate_record_AP_side = zeros(learning_Step_Max, 1);
Q_est_from_AP_side = zeros(total_simulated_CH_num, 1);
Q_est_from_STA_side = zeros(total_simulated_CH_num, 1);
% others
File_idx_pre = 0;
if_terminate_training = 0;
ep_idx = 1;
while 1
    if ep_idx <= parameters.num_episode_per_CASE
        ep_idx = ep_idx + 1;
    else
        ep_idx = 1;
    end
    %% two con-current Channel-related information feedback
    snap_idx = 1;
    while 1
        %% Break condition of inner loop
        if snap_idx > parameters.snap_total_num_for_test
            break;
        end
        if return_val_index > collect_data_num
            if_terminate_training = 1;
            break;
        end
        %% determine CSI comprssion mode for real-time channel sounding procedure
        if CH_sounding_idx == 1
            CSI_compression_mode_idx = 16;
            action_record_AP_side(CH_sounding_idx, 1) = CSI_compression_mode_idx;
        end
        % determine parameters of CSI compression mode
        CSI_compression_modes;
        %% load CSI data generated by COST2100
        feedback_interval_in_snap = feedback_interval/parameters.snap_interval;
        File_idx = ceil(snap_idx/parameters.snap_num_per_second);
        if File_idx ~= File_idx_pre
            if_load_file = 1;
        else
            if_load_file = 0;
        end
        File_idx_pre = File_idx;
        if if_load_file
            openfile = sprintf('/Case_%d_episode_%d_File_idx_%d_CSI_data.mat', ...
                parameters.CASE_IDX, ep_idx, File_idx);
            file_open_path = strcat(parameters.CSI_data_for_performance_test_saving_path, openfile);
            if exist(file_open_path,'file') == 0
                break;
            end
            load(file_open_path, 'h_CFR_store', 'h_CIR_store');
        end
        if mod(snap_idx, parameters.snap_num_per_second) ~= 0
            H_CFR = h_CFR_store{mod(snap_idx, parameters.snap_num_per_second), 1};
            H_CIR = h_CIR_store{mod(snap_idx, parameters.snap_num_per_second), 1};
        else
            H_CFR = h_CFR_store{parameters.snap_num_per_second, 1};
            H_CIR = h_CIR_store{parameters.snap_num_per_second, 1};
        end
        % convert format of original h_CFR
        h_CFR_3D_mat = zeros(parameters.STA_antenna_num, parameters.AP_antenna_num, parameters.occupied_subcarrier);
        for x = 1 : parameters.AP_antenna_num
            for y = 1 : parameters.STA_antenna_num
                h_CFR_3D_mat(y, x, :) = H_CFR(:, y, x);
            end % end of for y
        end % end of for x
        % obtain CSI at the end of feedback interval
        if feedback_interval_in_snap ~= 1
            snap_idx_End_of_CH = snap_idx + feedback_interval_in_snap - 1;
            if snap_idx_End_of_CH > parameters.snap_total_num
                % break of inner loop
                break;
            end
            File_idx_End_of_CH = ceil(snap_idx_End_of_CH/parameters.snap_num_per_second);
            if mod(snap_idx_End_of_CH, parameters.snap_num_per_second) == 1
                openfile = sprintf('/Case_%d_episode_%d_File_idx_%d_CSI_data.mat', ...
                    parameters.CASE_IDX, ep_idx, File_idx_End_of_CH);
                file_open_path = strcat(parameters.CSI_data_for_performance_test_saving_path, openfile);
                if exist(file_open_path,'file') == 0
                    break;
                end
                load(file_open_path, 'h_CFR_store');
            end
            if mod(snap_idx_End_of_CH, parameters.snap_num_per_second) ~= 0
                H_CFR_end_of_CH = h_CFR_store{mod(snap_idx_End_of_CH, parameters.snap_num_per_second), 1};
            else
                H_CFR_end_of_CH = h_CFR_store{parameters.snap_num_per_second, 1};
            end
            % convert format of original h_CFR
            h_CFR_3D_mat_End_of_CH = zeros(parameters.STA_antenna_num, parameters.AP_antenna_num, parameters.occupied_subcarrier);
            for x = 1 : parameters.AP_antenna_num
                for y = 1 : parameters.STA_antenna_num
                    h_CFR_3D_mat_End_of_CH(y, x, :) = H_CFR_end_of_CH(:, y, x);
                end % end of for y
            end % end of for x
        else
            h_CFR_3D_mat_End_of_CH = h_CFR_3D_mat;
        end
        %% Exectue channel soudning procedure
        if CSI_compression_mode_idx <= parameters.num_of_FD_CSI_compression_modes
            %% Frequecny domain CSI feedback
            % Step 1: spatial compression
            [phi_mat, psi_mat, idx_phi, idx_psi, num_psi_phi] = ...
                spatial_compression(h_CFR_3D_mat, parameters.occupied_subcarrier, ...
                parameters.AP_antenna_num, parameters.STA_antenna_num);
            % Step 2: angle quantization
            [phi_mat_quants, psi_mat_quants] = ...
                angle_quantization(phi_mat, psi_mat, parameters.occupied_subcarrier, ...
                q_phi, q_psi);
            % Step 3: tone grouping
            [phi_mat_fb, psi_mat_fb] = ...
                tone_grouping(phi_mat_quants, psi_mat_quants, parameters.occupied_subcarrier, N_g);
        end
        %% Calculate Channel Feature-related Information (CFI)
        [H_a_d] = angle_delay_transform(H_CFR, parameters.occupied_subcarrier,...
            parameters.AP_antenna_num, parameters.STA_antenna_num, parameters.N_c);
        % (1) calculate PDP in the paper
        [PDP] = PDP_cal(H_a_d, parameters.STA_antenna_num, parameters.N_c);
        CF{1, 1} = PDP;
        % (2) calculate PAS in the paper
        [PAS] = PAS_cal(H_a_d, parameters.AP_antenna_num, parameters.STA_antenna_num);
        CF{2, 1} = PAS;
        % (3) calculate Channel varation level
        if CH_sounding_idx > 1
            [v, v_vec] = channel_varation_level_cal(H_a_d_pre, H_a_d, parameters.STA_antenna_num);
        else
            v = 0;
            v_vec = zeros(parameters.STA_antenna_num, 1);
        end
        CF{3, 1} = v_vec;
        H_a_d_pre = H_a_d;
        %% Calculate the error of reconstructed CSI
        if CSI_compression_mode_idx <= parameters.num_of_FD_CSI_compression_modes
            %% Reconstruct CSI based on Frequecny domain CSI feedback
            % reconstruct real-time CSI
            [V_recon_3D_mat] = reconstruction(phi_mat_fb, psi_mat_fb, parameters.occupied_subcarrier, ...
                parameters.AP_antenna_num, parameters.STA_antenna_num);
            % calculate eeror of reconstructed CSI (compare estimated CSI with real CSI at the end of channel sounding)
            [nmse] = accuracy_CSI_fb(V_recon_3D_mat, h_CFR_3D_mat_End_of_CH, parameters.occupied_subcarrier,...
                parameters.AP_antenna_num, parameters.STA_antenna_num, 0);
            % Overhead calculation
            num_bits_of_BR = (parameters.occupied_subcarrier/N_g)*num_psi_phi*...
                (q_psi + q_phi);
            nmse_record(CH_sounding_idx) = nmse;
        end % end of i
        %% Calculation of airtime overhead
        num_bits_of_BR = num_bits_of_BR + parameters.BR_service_field + parameters.BR_MAC_header + ...
            parameters.BR_tail_bits;
        time_duration_BR = parameters.preamble_BR + (num_bits_of_BR/parameters.num_bits_per_OFDM_symbol)...
            * parameters.duration_OFDM_symbol;
        airtime_overhead = parameters.time_duration_NDPA + parameters.time_duration_NDP + ...
            time_duration_BR + 3*parameters.time_duration_SIFS;
        CH_airtime_overhead(CH_sounding_idx) = airtime_overhead;
        CH_overhead(CH_sounding_idx) = num_bits_of_BR;
        %% Oppturtinistic CFI feedback at STA side
        duration_of_two_CFI_fed = duration_of_two_CFI_fed + 1; % in number of channel sounding
        if CH_sounding_idx > 1
            if if_perform_validate
                learning_step_temp = learning_Step_Max;
            else
                learning_step_temp = learning_step;
            end
            fprintf('Snap_idx = %d: Start CFI feedback. \n', snap_idx);
            % STA decide which information to feedback based on its DQN
            [Positional_Encoding_vec1, Positional_Encoding_vec2, Positional_Encoding_vec3] = ...
                Positional_Encoding(CSI_compression_mode_idx, parameters.N_c, parameters.AP_antenna_num, parameters.STA_antenna_num);
            action_result_for_STA = pyrunfile('MARL/MARL_STA_part.py', "output_for_MATLAB",...
                episode_from_matlab = learning_step_temp,...
                cache_length_from_matlab = 2,...
                PDP_state_size_from_matlab = parameters.N_c,...
                PAS_state_size_from_matlab = parameters.AP_antenna_num, ...
                CSL_state_size_from_matlab = parameters.STA_antenna_num,...
                action_size_from_matlab = 8,...
                Root_Path_File_from_matlab = DNN_file_save_dir,...
                Operation_mode_from_matlab = 2,... % action mode
                PDP_state_seq_from_matlab = CF{1, 1} + Positional_Encoding_vec1, ...
                PAS_state_seq_from_matlab = CF{2, 1} + Positional_Encoding_vec2, ...
                CSL_state_seq_from_matlab = CF{3, 1} + Positional_Encoding_vec3, ...
                action_seq_from_matlab = zeros(1,2), reward_seq_from_matlab = zeros(1,2), ...
                next_PDP_state_seq_from_matlab = zeros(parameters.N_c, 1),...
                next_PAS_state_seq_from_matlab = zeros(parameters.AP_antenna_num, 1), ...
                next_CSL_state_seq_from_matlab = zeros(parameters.STA_antenna_num, 1));
            action_result_for_STA = struct(action_result_for_STA);
            CFI_type_index = double(action_result_for_STA.action_idx);
            state_action_values_STA = double(action_result_for_STA.state_action_values);
            state_action_values_STA_record(CH_sounding_idx, :) = state_action_values_STA';
            Q_est_from_STA_side(CH_sounding_idx, 1) = ...
                state_action_values_STA_record(CH_sounding_idx, CFI_type_index);
            action_record_STA_side(CH_sounding_idx, 1) = CFI_type_index;
            CF_record(CH_sounding_idx, :) = [CF{1, 1} + Positional_Encoding_vec1; CF{2, 1} + Positional_Encoding_vec2; ...
                CF{3, 1} + Positional_Encoding_vec3]';
            if CH_sounding_idx >= 3 && ~if_perform_validate
                Memory_store(CH_sounding_idx - 1, 'STA')
            end
            CF_real_record(CH_sounding_idx, :) = [CF{1, 1}; CF{2, 1}; CF{3, 1}]';
            CFI{1, 1} = zeros(parameters.N_c, 1);
            CFI{2, 1} = zeros(parameters.AP_antenna_num, 1);
            CFI{3, 1} = zeros(parameters.STA_antenna_num, 1);
            data_size_CFI = 0;
            if CFI_type_index ~= 8  % CFI_type_index ~= 8 means feedback feature information
                if_trigger_CFI_Fed = 1;
                switch CFI_type_index
                    case 1
                        CFI{1, 1} = CF{1, 1};
                        data_size_CFI = parameters.N_c*32;
                    case 2
                        CFI{2, 1} = CF{2, 1};
                        data_size_CFI = parameters.AP_antenna_num*32;
                    case 3
                        CFI{3, 1} = CF{3, 1};
                        data_size_CFI = parameters.STA_antenna_num*32;
                    case 4
                        CFI{1, 1} = CF{1, 1};
                        CFI{2, 1} = CF{2, 1};
                        data_size_CFI = parameters.N_c*32 + parameters.AP_antenna_num*32;
                    case 5
                        CFI{1, 1} = CF{1, 1};
                        CFI{3, 1} = CF{3, 1};
                        data_size_CFI = parameters.N_c*32 + parameters.STA_antenna_num*32;
                    case 6
                        CFI{2, 1} = CF{2, 1};
                        CFI{3, 1} = CF{3, 1};
                        data_size_CFI = parameters.AP_antenna_num*32 + parameters.STA_antenna_num*32;
                    case 7
                        CFI{1, 1} = CF{1, 1};
                        CFI{2, 1} = CF{2, 1};
                        CFI{3, 1} = CF{3, 1};
                        data_size_CFI = parameters.N_c*32 + parameters.AP_antenna_num*32 + parameters.STA_antenna_num*32;
                end % end of switch
            else % CFI_type_index = 0 means feedback nothing
                if_trigger_CFI_Fed = 0;
            end % end of "if CFI_type_index > 0"
            CFI_overhead(CH_sounding_idx, 1) = data_size_CFI;
        end % end of if CFI_Fed
        %% Calculation of utlity(reward) function
        utility_value_per_sounding = util_fun_nmse(nmse,parameters.nmse_in_dB_low_bound,parameters.nmse_in_dB_up_bound, ...
            parameters.nmse_util_TH_value) + ...
            parameters.rho_0*util_fun_CH_overhead(airtime_overhead, feedback_interval, parameters.AO_max, ...
            parameters.AO_min);
        utility_value_per_sounding_record(CH_sounding_idx, 1) = utility_value_per_sounding;
        if CH_sounding_idx > 1
            reward_fun_record_STA_side(CH_sounding_idx - 1, 1) = utility_value_per_sounding - ...
                parameters.rho_1*util_fun_CFI_overhead(CFI_overhead(CH_sounding_idx - 1, 1), parameters.CFI_data_size_max);
            reward_fun_record_AP_side(CH_sounding_idx - 1, 1) = utility_value_per_sounding - ...
                parameters.rho_1*util_fun_CFI_overhead(CFI_overhead(CH_sounding_idx - 1, 1), parameters.CFI_data_size_max);
        end
        %% check
        nmse_util_record(CH_sounding_idx, 1) = nmse;
        nmse_util_record(CH_sounding_idx, 2) = 10*log10(nmse);
        nmse_util_record(CH_sounding_idx, 3) = util_fun_nmse(nmse,parameters.nmse_in_dB_low_bound,parameters.nmse_in_dB_up_bound, ...
            parameters.nmse_util_TH_value);
        CH_overhead_util(CH_sounding_idx, 1) = CSI_compression_mode_idx;
        CH_overhead_util(CH_sounding_idx, 2) = airtime_overhead;
        CH_overhead_util(CH_sounding_idx, 3) = feedback_interval;
        CH_overhead_util(CH_sounding_idx, 4) = airtime_overhead/feedback_interval;
        CH_overhead_util(CH_sounding_idx, 5) = util_fun_CH_overhead(airtime_overhead, feedback_interval, parameters.AO_max, ...
            parameters.AO_min);
        %% New CFI feedback
        CFI_fedback_idx = CFI_fedback_idx + 1;
        %% Make decisions to adapt the channel sounding at AP side based on DRL
        if CH_sounding_idx > 1
            if if_perform_validate
                learning_step_temp = learning_Step_Max;
            else
                learning_step_temp = learning_step;
            end
            [Positional_Encoding_vec1, Positional_Encoding_vec2, Positional_Encoding_vec3] = ...
                Positional_Encoding(CSI_compression_mode_idx, parameters.N_c, parameters.AP_antenna_num, parameters.STA_antenna_num);
            fprintf('Snap_idx = %d: Start Adaption of CSI feedback. \n', snap_idx);
            action_result_for_AP = pyrunfile('MARL/MARL_AP_part.py',...
                "output_for_MATLAB", episode_from_matlab = learning_step_temp,...
                cache_length_from_matlab = 1,...
                PDP_state_size_from_matlab = parameters.N_c,...
                PAS_state_size_from_matlab = parameters.AP_antenna_num, ...
                CSL_state_size_from_matlab = parameters.STA_antenna_num,...
                action_size_from_matlab = parameters.num_of_CSI_compression_modes, ...
                Root_Path_File_from_matlab = DNN_file_save_dir,...
                Operation_mode_from_matlab = 1,... % action mode
                PDP_state_seq_from_matlab = CFI{1, 1} + Positional_Encoding_vec1, ...
                PAS_state_seq_from_matlab = CFI{2, 1} + Positional_Encoding_vec2, ...
                CSL_state_seq_from_matlab = CFI{3, 1} + Positional_Encoding_vec3, ...
                action_seq_from_matlab = zeros(1,2), reward_seq_from_matlab = zeros(1,2), ...
                next_PDP_state_seq_from_matlab = zeros(parameters.N_c, 1),...
                next_PAS_state_seq_from_matlab = zeros(parameters.AP_antenna_num, 1), ...
                next_CSL_state_seq_from_matlab = zeros(parameters.STA_antenna_num, 1));
            action_result_for_AP = struct(action_result_for_AP);
            state_action_values_AP = double(action_result_for_AP.state_action_values);
            state_action_values_AP_record(CH_sounding_idx, :) = state_action_values_AP';
            if if_trigger_CFI_Fed
                CSI_compression_mode_idx = double(action_result_for_AP.action_idx);
                Q_est_from_AP_side(CH_sounding_idx, 1) = ...
                    state_action_values_AP_record(CH_sounding_idx, ...
                    CSI_compression_mode_idx);
            end
            action_record_AP_side(CH_sounding_idx, 1) = CSI_compression_mode_idx;
            CFI_record(CH_sounding_idx, :) = [CFI{1, 1} + Positional_Encoding_vec1; ...
                CFI{2, 1} + Positional_Encoding_vec2; CFI{3, 1} + Positional_Encoding_vec3]';
            if CH_sounding_idx >= 3 && ~if_perform_validate
                % when performing validation, we not collect data
                Memory_store(CH_sounding_idx - 1, 'AP');
            end
        end % end of if CH_sounding_idx > 1
        %% -----------------------Validation Process------------------------
        if if_perform_validate && CH_sounding_idx > 3
            if validation_sample_num >= trajectory_len || ...
                    snap_idx + feedback_interval_in_snap > parameters.snap_total_num_for_test
                % end of validation process
                if validation_sample_num >= trajectory_len
                    return_val_index = return_val_index + 1;
                end
                if validation_sample_num < trajectory_len
                    return_val_discount_record(return_val_index, 1) = 0;
                    Q_est_from_STA_side_versus_return(return_val_index, 1) = ...
                        0;
                    Q_est_from_AP_side_versus_return(return_val_index, 1) = ...
                        0;
                end
                validation_sample_num = 0;
            else
                if validation_sample_num == 0
                    Q_est_from_STA_side_versus_return(return_val_index, 1) = ...
                        Q_est_from_STA_side(CH_sounding_idx - 1, 1);
                    Q_est_from_AP_side_versus_return(return_val_index, 1) = ...
                        Q_est_from_AP_side(CH_sounding_idx - 1, 1);
                end
                return_val_discount_record(return_val_index, 1) = ...
                    return_val_discount_record(return_val_index, 1) + ...
                    reward_fun_record_AP_side(CH_sounding_idx - 1, 1)*...
                    ((discount_factor)^(validation_sample_num));
                validation_sample_num = validation_sample_num + 1;
            end
        end
        %% Check Point
        if mod(learning_step, 400) == 0
            if learning_step > 10
                figure(1)
                subplot(2,1,1)
                plot(1:learning_step, LOSS_record_AP_side(1:learning_step),'-');
                hold on
                plot(1:learning_step, LOSS_record_STA_side(1:learning_step),'--');
                legend('Training loss of AP','Training loss of STA');
                hold off
                subplot(2,1,2)
                plot(1:learning_step, learning_rate_record_AP_side(1:learning_step),'-');
                hold on
                plot(1:learning_step, learning_rate_record_STA_side(1:learning_step),'--');
                legend('learning rate of AP','learning rate of STA');
                hold off
                figure(2)
                subplot(2,1,1)
                histogram(action_record_AP_side(4:CH_sounding_idx))
                hold on
                histogram(action_record_STA_side(4:CH_sounding_idx))
                legend('Actions of AP','Actions of STA');
                hold off
                pause(1);
            end
        end % end of check point
        %% Renew Channel Sounding Index
        CH_sounding_idx = CH_sounding_idx + 1;
        %% Renew snap index
        snap_idx = snap_idx + feedback_interval_in_snap;
    end % end of loop_snap
    %% break condtion of outer loop
    if if_terminate_training
        break
    end
    %% save files
    if mod(CH_soundinthg_idx, 10) == 0
        savefile = sprintf('/Case_%d_CSI_fed_result_DQN_validation_Q_est_CASE_%d.mat', parameters.CASE_IDX, training_CASE);
        file_save_path = strcat(file_save_dir, savefile);
        save(file_save_path, ...
            'learning_step','ep_idx','CH_sounding_idx',...
            'reward_fun_record_AP_side', 'reward_fun_time_average_record_AP_side',...
            'LOSS_record_AP_side', 'LOSS_record_STA_side','utility_value_per_sounding_record','CH_airtime_overhead','CH_overhead',...
            'CH_overhead_util','CFI_overhead','action_record_AP_side','action_record_STA_side','nmse_record',...
            'learning_rate_record_AP_side', 'learning_rate_record_STA_side',...
            'state_action_values_AP_record', 'state_action_values_STA_record',...
            'return_val_discount_record',...
            'Q_est_from_STA_side_versus_return','Q_est_from_STA_side',...
            'Q_est_from_AP_side_versus_return', 'Q_est_from_AP_side');
    end
    %% Reset Channel Sounding Index when inner loop is terminated
    CH_sounding_idx = CH_sounding_idx - 1;
    ep_idx = ep_idx + 1;
end % for ep_idx