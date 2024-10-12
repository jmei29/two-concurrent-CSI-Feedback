# Overview
This is the Matlab and PyTorch implementation of the paper "Learning Aided Closed-loop Feedback: A Concurrent Dual Channel Information Feedback Mechanism for Wi-Fi". If you feel this repo helpful, please cite our paper:
```
@article{Jie2024,
  title={Learning Aided Closed-loop Feedback: A Concurrent Dual Channel Information Feedback Mechanism for {Wi-Fi}},
  author={Jie, Mei，Xianbin, Wang， and Kan Zheng},
  journal={IEEE TWC},
  year={2024, Early Access}
}
```
# Requirements
To implement this project, you need to ensure the following requirements are installed.
 * Matlab >= 2021b
 * Python = 3.8 or 3.9, please refere to [Versions of Python Compatible with MATLAB Products by Release](https://www.mathworks.com/support/requirements/python-compatibility.html)
 * Pytorch >= 1.2

# Project Preparation

## Channel generation
The channel state information (CSI) matrix is generated from COST2100 model. You can generate your own dataset according to the open source library of [COST2100](https://github.com/cost2100/cost2100) as well. The details of data pre-processing can be found in our paper.

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
├── CSI data set  # The data folder
...
```
# Contact
If you have any problem with this code, please feel free to contact meijie@nbu.eud.cn.

