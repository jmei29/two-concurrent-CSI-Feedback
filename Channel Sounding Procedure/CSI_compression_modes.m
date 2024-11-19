
switch CSI_compression_mode_idx
    %% frequency-domain CSI compression mode
    % elemets in each row
    % 1. the feedback interval between two successive channel sounding
    % 2. the number of tones in a group
    % 3 and 4. the number of bits for quantizing angles in
    %          $\boldsymbol{\Psi}$ (\in [0, \pi/2])
    %          and $\boldsymbol{\Phi}$ (\in [0, 2*\pi]), respectively.
    case 1
        feedback_interval = 1*parameters.snap_interval; % in millseconds
        N_g = 1;
        q_psi = 7;
        q_phi = 9;
    case 2
        feedback_interval = 1*parameters.snap_interval; % in millseconds
        N_g = 4;
        q_psi = 7;
        q_phi = 9;
    case 3
        feedback_interval = 1*parameters.snap_interval; % in millseconds
        N_g = 8;
        q_psi = 7;
        q_phi = 9;
    case 4
        feedback_interval = 1*parameters.snap_interval; % in millseconds
        N_g = 16;
        q_psi = 7;
        q_phi = 9;
    case 5
        feedback_interval = 2*parameters.snap_interval; % in millseconds
        N_g = 1;
        q_psi = 5;
        q_phi = 7;
    case 6
        feedback_interval = 2*parameters.snap_interval; % in millseconds
        N_g = 4;
        q_psi = 5;
        q_phi = 7;
    case 7
        feedback_interval = 2*parameters.snap_interval; % in millseconds
        N_g = 8;
        q_psi = 5;
        q_phi = 7;
    case 8
        feedback_interval = 2*parameters.snap_interval; % in millseconds
        N_g = 16;
        q_psi = 5;
        q_phi = 7;
    case 9
        feedback_interval = 3*parameters.snap_interval; % in millseconds
        N_g = 1;
        q_psi = 7;
        q_phi = 9;
    case 10
        feedback_interval = 3*parameters.snap_interval; % in millseconds
        N_g = 4;
        q_psi = 7;
        q_phi = 9;
    case 11
        feedback_interval = 3*parameters.snap_interval; % in millseconds
        N_g = 8;
        q_psi = 7;
        q_phi = 9;
    case 12
        feedback_interval = 3*parameters.snap_interval; % in millseconds
        N_g = 16;
        q_psi = 7;
        q_phi = 9;
    case 13
        feedback_interval = 4*parameters.snap_interval; % in millseconds
        N_g = 1;
        q_psi = 5;
        q_phi = 7;
    case 14
        feedback_interval = 4*parameters.snap_interval; % in millseconds
        N_g = 4;
        q_psi = 5;
        q_phi = 7;
    case 15
        feedback_interval = 4*parameters.snap_interval; % in millseconds
        N_g = 8;
        q_psi = 5;
        q_phi = 7;
    case 16
        feedback_interval = 4*parameters.snap_interval; % in millseconds
        N_g = 16;
        q_psi = 5;
        q_phi = 7;
end % end of switch
