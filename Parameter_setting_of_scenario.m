clc
clear
%% Define the index of simulation scenaros
parameters.CASE_IDX = 1;
%% Chosen CASE
% In COST 2100, Choose a surrounding radio enviroment type of
% {'IndoorHall_5GHz','SemiUrban_300MHz','Indoor_CloselySpacedUser_2_6GHz',...
% 'SemiUrban_CloselySpacedUser_2_6GHz','SemiUrban_VLA_2_6GHz'}
Data_saving_Root_dir = '/Volumes/DATA/CSI_feedback';
parameters.Wi_Fi_settings = 6;
% CSI Data Saving
CASE_filename = sprintf('/CSI-data-set/Case_%d', parameters.CASE_IDX);
CASE_file_path = strcat(Data_saving_Root_dir,CASE_filename);
if ~exist(CASE_file_path, 'dir')
    mkdir(CASE_file_path)
end
parameters.CSI_data_saving_path = CASE_file_path; % CSI data saving path
% Test data savin
Performance_Test_filename = sprintf('/CSI-data-set/Case_%d/Test', parameters.CASE_IDX);
Test_CASE_file_path = strcat(Data_saving_Root_dir, Performance_Test_filename);
if ~exist(Test_CASE_file_path, 'dir')
    mkdir(Test_CASE_file_path)
end
parameters.CSI_data_for_performance_test_saving_path = Test_CASE_file_path; % CSI data saving path
% Training Data Saving
Training_Data_filename = sprintf('/Train_data/Case_%d', parameters.CASE_IDX);
Training_Data_file_path = strcat(Data_saving_Root_dir, Training_Data_filename);
if ~exist(Training_Data_file_path, 'dir')
    mkdir(Training_Data_file_path)
end
parameters.CSI_train_data_saving_path = Training_Data_file_path;
% Result Data Saving
CASE_result_filename = sprintf('/Result/Case_%d', parameters.CASE_IDX);
CASE_result_file_path = strcat(Data_saving_Root_dir, CASE_result_filename);
if ~exist(CASE_result_file_path, 'dir')
    mkdir(CASE_result_file_path)
end
parameters.result_data_saving_path = CASE_result_file_path; % CSI data saving path
% DNN Saving
DNN_saving_Root_dir = '/Users/jiemei/Documents/GitHub/CSI-Feedback-for-Next-Generation-Wi-Fi/Channel Sounding Procedure';
CASE_DNN_filename = sprintf('/MARL_DNN/Case_%d', parameters.CASE_IDX);
CASE_DNN_file_path = strcat(DNN_saving_Root_dir, CASE_DNN_filename);
if ~exist(CASE_DNN_file_path, 'dir')
    mkdir(CASE_DNN_file_path)
end
parameters.DNN_saving_path = CASE_DNN_file_path;
parameters.duration_OFDM_symbol = 4/1000; % time duration for OFDM symbol, in millseconds
parameters.time_duration_NDPA = (20 + (16 + 168 + 32 +18)/29)/1000; % in millseconds
% ﻿all control frames duplicated in every 20 MHz sub-channel when wider channels are used in a single spatial stream
parameters.time_duration_NDP = 168/1000; % in millseconds
parameters.time_duration_SIFS = 16/1000; % in millseconds
parameters.preamble_BR = 164/1000; % time duration of preamble of BR frame, in millseconds
% Overhead of BR frame
parameters.BR_service_field = 32; % in bits
parameters.BR_MAC_header = 320; % in bits
parameters.BR_tail_bits = 18; % in bits
switch parameters.Wi_Fi_settings
    case 0
        parameters.bandwidth = 20*1e6; % 20MHz
        parameters.AP_antenna_num = 8; % number of antennas in the AP side
        parameters.STA_antenna_num = 2; % number of antennas in the STA side
        parameters.antenna_type = 'MIMO_dipole'; % Choose an Antenna type out of
        % {'MIMO_dipole', 'MIMO_measured'}
        parameters.LOS_or_NLOS = 'LOS';
        parameters.radio_env_type = 'IndoorHall_5GHz';
        parameters.num_bits_per_OFDM_symbol = floor(14.6*4); % number of bits per OFDM symbol
    case 1
        parameters.bandwidth = 20*1e6; % 20MHz
        parameters.AP_antenna_num = 16; % number of antennas in the AP side
        parameters.STA_antenna_num = 2; % number of antennas in the STA side
        parameters.antenna_type = 'MIMO_dipole'; % Choose an Antenna type out of
        % {'MIMO_dipole', 'MIMO_measured'}
        parameters.LOS_or_NLOS = 'LOS';
        parameters.radio_env_type = 'IndoorHall_5GHz';
        parameters.num_bits_per_OFDM_symbol = floor(14.6*4); % number of bits per OFDM symbol
    case 2
        parameters.bandwidth = 40*1e6; % 40MHz
        parameters.AP_antenna_num = 8; % number of antennas in the AP side
        parameters.STA_antenna_num = 2; % number of antennas in the STA side
        parameters.antenna_type = 'MIMO_dipole'; % Choose an Antenna type out of
        % {'MIMO_dipole', 'MIMO_measured'}
        parameters.LOS_or_NLOS = 'LOS'; % LOS or NLOS
        parameters.radio_env_type = 'IndoorHall_5GHz';
        parameters.num_bits_per_OFDM_symbol = floor(14.6*4); % number of bits per OFDM symbol
    case 3
        parameters.bandwidth = 40*1e6; % 40MHz
        parameters.AP_antenna_num = 16; % number of antennas in the AP side
        parameters.STA_antenna_num = 2; % number of antennas in the STA side
        parameters.antenna_type = 'MIMO_dipole'; % Choose an Antenna type out of
        % {'MIMO_dipole', 'MIMO_measured'}
        parameters.LOS_or_NLOS = 'LOS'; % LOS or NLOS
        parameters.radio_env_type = 'IndoorHall_5GHz';
        parameters.num_bits_per_OFDM_symbol = floor(14.6*4); % number of bits per OFDM symbol
    case 4
        parameters.bandwidth = 80*1e6; % 80MHz
        parameters.AP_antenna_num = 8; % number of antennas in the AP side
        parameters.STA_antenna_num = 2; % number of antennas in the STA side
        parameters.antenna_type = 'MIMO_dipole'; % Choose an Antenna type out of
        % {'MIMO_dipole', 'MIMO_measured'}
        parameters.LOS_or_NLOS = 'LOS'; % LOS or NLOS
        parameters.radio_env_type = 'IndoorHall_5GHz';
        parameters.num_bits_per_OFDM_symbol = floor(30.6*4); % number of bits per OFDM symbol
    case 5
        parameters.bandwidth = 80*1e6; % 80MHz
        parameters.AP_antenna_num = 16; % number of antennas in the AP side
        parameters.STA_antenna_num = 2; % number of antennas in the STA side
        parameters.antenna_type = 'MIMO_dipole'; % Choose an Antenna type out of
        % {'MIMO_dipole', 'MIMO_measured'}
        parameters.LOS_or_NLOS = 'LOS'; % LOS or NLOS
        parameters.radio_env_type = 'IndoorHall_5GHz';
        parameters.num_bits_per_OFDM_symbol = floor(30.6*4); % number of bits per OFDM symbol
    case 6
        parameters.bandwidth = 160*1e6; % 160MHz
        parameters.AP_antenna_num = 8; % number of antennas in the AP side
        parameters.STA_antenna_num = 2; % number of antennas in the STA side
        parameters.antenna_type = 'MIMO_dipole'; % Choose an Antenna type out of
        % {'MIMO_dipole', 'MIMO_measured'}
        parameters.LOS_or_NLOS = 'LOS'; % LOS or NLOS
        parameters.radio_env_type = 'IndoorHall_5GHz';
        parameters.num_bits_per_OFDM_symbol = floor(61.2*4); % number of bits per OFDM symbol
    case 7
        parameters.bandwidth = 160*1e6; % 160MHz
        parameters.AP_antenna_num = 16; % number of antennas in the AP side
        parameters.STA_antenna_num = 2; % number of antennas in the STA side
        parameters.antenna_type = 'MIMO_dipole'; % Choose an Antenna type out of
        % {'MIMO_dipole', 'MIMO_measured'}
        parameters.LOS_or_NLOS = 'LOS'; % LOS or NLOS
        parameters.radio_env_type = 'IndoorHall_5GHz';
        parameters.num_bits_per_OFDM_symbol = floor(61.2*4); % number of bits per OFDM symbol
end
%% Setting of Communication Systems
parameters.center_carrier_freq = 5e9; % 5GHz
parameters.freq_start = parameters.center_carrier_freq - parameters.bandwidth/2;
parameters.freq_end = parameters.center_carrier_freq + parameters.bandwidth/2;
parameters.subcarrier_spacing = 78.125e3; % 78.125 kHz
switch parameters.bandwidth
    case 20*1e6
        parameters.FFT_length = 256;
        parameters.occupied_subcarrier = 256;
    case 40*1e6
        parameters.FFT_length = 512;
        parameters.occupied_subcarrier = 512;
    case 80*1e6
        parameters.FFT_length = 1024;
        parameters.occupied_subcarrier = 1024;
    case 160*1e6
        parameters.FFT_length = 2048;
        parameters.occupied_subcarrier = 2048;
end
parameters.Tb = 1/parameters.subcarrier_spacing;
parameters.Tg = 0.8e-6;% in microsecond
% parameters.Tg = parameters.Tb*(parameters.CP_length/parameters.FFT_length);
parameters.CP_length = parameters.FFT_length*(parameters.Tg/parameters.Tb);
parameters.sampling_duration = 1/(parameters.subcarrier_spacing*(parameters.FFT_length + parameters.CP_length));
parameters.sampling_freq = 1/parameters.sampling_duration;
parameters.OFDM_symbol_duration = parameters.Tb + parameters.Tg;
parameters.OFDM_symbol_length = parameters.FFT_length + parameters.CP_length;
parameters.large_scale_fading_on_off = 0;
% 1: there is large-scale fading; 0: otherwises.
% parameters.occupied_subcarrier_index = [parameters.CP_length+(parameters.FFT_length-parameters.occupied_subcarrier)/2:...
%    parameters.CP_length+(parameters.FFT_length-parameters.occupied_subcarrier)/2+parameters.occupied_subcarrier/2-1,...
%    parameters.CP_length+(parameters.FFT_length-parameters.occupied_subcarrier)/2+parameters.occupied_subcarrier/2+1:...
%    parameters.CP_length+(parameters.FFT_length-parameters.occupied_subcarrier)/2+parameters.occupied_subcarrier]';
%% Network Parameters: only one AP and one user STA is considered
parameters.AP_pos = [0 0 3]; % Center position of BS array [x, y, z] [m]
parameters.STA_AP_2d_distance = 15; % in meters
parameters.STA_Velo = [0.05 0 0]; % [m/s]
%% Data set size
parameters.num_episode_per_CASE = 30; % number of episodes per CASE
parameters.snap_interval = 100; % in millseconds
parameters.snap_num_per_second = 1000/parameters.snap_interval; % Number of snapshots per second
parameters.total_simulation_duration = 20; % in seconds
parameters.snap_total_num = parameters.total_simulation_duration*parameters.snap_num_per_second; % Total number of snapshots
parameters.total_simulation_duration_for_test = parameters.total_simulation_duration/4; % in seconds
parameters.snap_total_num_for_test = parameters.total_simulation_duration_for_test*parameters.snap_num_per_second;
% Total number of snapshots for test
%% Angular-Delay domain transformation
parameters.N_c = parameters.FFT_length/8;
%% Trigger Condition
parameters.TH_PDP = 0.05;
parameters.TH_PAS = 0.05;
parameters.TH_v = 0.05;
%% Time domain CSI compression mode
parameters.epsilon = 0;
%% Number of CSI compression modes
parameters.num_of_FD_CSI_compression_modes = 16;
parameters.num_of_TD_CSI_compression_modes = 0;
parameters.num_of_CSI_compression_modes = ...
    parameters.num_of_FD_CSI_compression_modes + parameters.num_of_TD_CSI_compression_modes;
parameters.CFI_size = parameters.N_c + parameters.AP_antenna_num + 1;
%% Definition of Utility Function
parameters.rho_0 = 0.5;
parameters.rho_1 = 0.2;
parameters.nmse_in_dB_up_bound = -5;
parameters.nmse_in_dB_low_bound = -15;
parameters.nmse_util_TH_value = 0.2;
parameters.AO_max = 0.07;
parameters.AO_min = 0.0026;
parameters.CFI_data_size_max = parameters.N_c*32 + parameters.AP_antenna_num*32 + parameters.STA_antenna_num*32;
%% SAVE parameter settings
savefile = sprintf('simualtion_parameters_CASE_%d.mat', parameters.CASE_IDX);
save(savefile, 'parameters');