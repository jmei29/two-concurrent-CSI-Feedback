# Overview
This is the Matlab and PyTorch implementation of the paper "Learning Aided Closed-loop Feedback: A Concurrent Dual Channel Information Feedback Mechanism for Wi-Fi". If you feel this repo helpful, please cite our paper:
```
@ARTICLE{10740600,
  author={Mei, Jie and Wang, Xianbin and Zheng, Kan},
  journal={IEEE Transactions on Wireless Communications}, 
  title={Learning Aided Closed-loop Feedback: A Concurrent Dual Channel Information Feedback Mechanism for {Wi-Fi}}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TWC.2024.3481054}}

```
# Requirements
To implement this project, you need to ensure the following requirements are installed.
 * Matlab >= 2021b
 * Python = 3.8 or 3.9, please refere to [Versions of Python Compatible with MATLAB Products by Release](https://www.mathworks.com/support/requirements/python-compatibility.html)
 * Pytorch >= 1.2

# Project Preparation

## Project Tree Arrangement
We recommend you to arrange the project tree as follows.

```
home # The cloned repository of "two-concurrent-CSI-Feedback"
├── Parameter_setting_of_scenario.m # Setting Parameters for simulation
├── Channel Sounding Procedure  
│   ├── MARL
│   │     ├── MARL_AP_PER_train_v2.py
│   │     ├── MARL_AP_part.py
│   │     ├── MARL_STA_part.py
│   │     ├── MARL_STA_PER_train_v2.py
│   ├── MARL_DNN # saving DNN model of Agents
│   ├── main_MARL_PER_train_v3_3_2.m
│   ├── main_MARL_for_test_v2.m
│   ├── main_MARL_PER_train_validate_Q_Est.m
│   ├── SumTree_update.m
│   ├── SumTree_total_p.m
│   ├── SumTree_get_leaf.m
│   ├── SumTree_add.m
│   ├── Memory_store.m
│   ├── Memory_sample.m
│   ├── Memory_batch_update.m
│   ├── CSI_compression_modes.m
│   ├── accuracy_psi_phi_fb.m
│   ├── accuracy_CSI_fb.m
│   ├── util_fun_nmse.m
│   ├── Compression_Idx_Encoding.m
│   ├── util_fun_CH_overhead.m
│   ├── util_fun_CFI_overhead.m
│   ├── tone_grouping.m
│   ├── spatial_compression.m
│   ├── reconstruction.m
│   ├── channel_varation_level_cal.m
│   ├── angle_quantization.m
│   ├── Positional_Encoding.m
│   ├── Result Analysis
│   │     ├── Fig_LOSS_Train.m
│   │     ├── Fig_Return_Curve.m
│   │     ├── Fig_Q_est_v3.m
│   │     ├── Fig2_Quantization_CDF_v2.m
│   │     ├── Fig3_CSI_Analysis_v2.m
├── CSI data set  # The data folder
...
```
# Run simulation
- CSI data generation: The channel state information (CSI) matrix is generated from COST2100 model. You can generate your own dataset according to the open source library of [COST2100](https://github.com/cost2100/cost2100) as well. The details of data pre-processing can be found in our paper.
- For detailed parameters, please refer to the "Parameter_setting_of_scenario.m" in the folder.
- Training: Run "main_MARL_PER_train_v3_3_2.m" in the folder of "Channel Sounding Procedure".
- Testing: Run "main_MARL_for_test_v2.m" and "main_MARL_PER_train_validate_Q_Est.m" in the folder of "Channel Sounding Procedure".

# Contact
If you have any problem with this code, please feel free to contact meijie@nbu.eud.cn.

