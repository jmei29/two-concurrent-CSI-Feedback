# Overview
This is the Matlab and PyTorch implementation of the paper "Learning to Feedback: A Two-concurrent Channel-related Information Feedback Mechanism for Next-Generation Wi-Fi". If you feel this repo helpful, please cite our paper:
```
@article{Jie2023,
  title={Learning to Feedback: A Two-concurrent Channel-related Information Feedback Mechanism for Next-Generation {Wi-Fi}},
  author={Jie, Mei and Xianbin, Wang},
  journal={},
  year={2023}
}
```
# Requirements
To implement this project, you need to ensure the following requirements are installed.
 * Matlab >= 2021b
 * Python >= 3.7
 * Pytorch >= 1.2

# Project Preparation

## Channel generation
The channel state information (CSI) matrix is generated from COST2100 model. You can generate your own dataset according to the open source library of COST2100 as well. The details of data pre-processing can be found in our paper.

## Project Tree Arrangement
