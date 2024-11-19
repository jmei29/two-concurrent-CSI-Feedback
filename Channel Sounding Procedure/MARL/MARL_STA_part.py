"""
import os
import sys

directory_matlab_folder = os.path.abspath(".")
module_file = r"{}/MARL".format(directory_matlab_folder)
module_path = sys.path.append(module_file)

from util import DRL_Agent, Net_STA
import numpy as np
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import random
from collections import deque
import numpy as np
import pickle as pkl
import copy
import os


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.fc1 = nn.Linear(2, 6)
        self.fc2 = nn.Linear(6, 1)

    def forward(self, x):
        x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        x_out_1 = F.relu(self.fc1(x_compress))
        x_out = F.relu(self.fc2(x_out_1))
        scale = torch.sigmoid(x_out)
        return x * scale


class Net_STA(nn.Module):
    def __init__(self, input_dim1, input_dim2, input_dim3, output_dim):
        super(Net_STA, self).__init__()

        self.spatial_gate1 = SpatialGate()
        # 输入张量1有1个通道
        self.spatial_gate2 = SpatialGate()
        # 输入张量2有1个通道
        self.spatial_gate3 = SpatialGate()
        # 输入张量3有1个通道

        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.input_dim3 = input_dim3
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim1 + self.input_dim2 + self.input_dim3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, self.output_dim)

    def forward(self, x1, x2, x3):
        x1_attended = self.spatial_gate1(x1)
        x2_attended = self.spatial_gate2(x2)
        x3_attended = self.spatial_gate3(x3)

        concatenated_output = torch.cat((x1_attended, x2_attended, x3_attended), dim=1)

        x = F.relu(self.fc1(concatenated_output))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return self.fc4(x)


class Net_STA_type_2(nn.Module):
    def __init__(self, input_dim1, input_dim2, input_dim3, input_dim4, output_dim):
        super(Net_STA_type_2, self).__init__()

        self.spatial_gate1 = SpatialGate()
        # 输入张量1有1个通道
        self.spatial_gate2 = SpatialGate()
        # 输入张量2有1个通道
        self.spatial_gate3 = SpatialGate()
        # 输入张量3有1个通道

        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.input_dim3 = input_dim3
        self.input_dim4 = input_dim4
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim1 + self.input_dim2 + self.input_dim3 + self.input_dim4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, self.output_dim)

    def forward(self, x1, x2, x3, x4):
        x1_attended = self.spatial_gate1(x1)
        x2_attended = self.spatial_gate2(x2)
        x3_attended = self.spatial_gate3(x3)

        concatenated_output = torch.cat((x1_attended, x2_attended, x3_attended, x4), dim=1)

        x = F.relu(self.fc1(concatenated_output))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return self.fc4(x)


class Net_AP(nn.Module):
    def __init__(self, input_dim1, input_dim2, input_dim3, output_dim):
        super(Net_AP, self).__init__()

        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.input_dim3 = input_dim3
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim1 + self.input_dim2 + self.input_dim3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, self.output_dim)

    def forward(self, x1, x2, x3):
        concatenated_output = torch.cat((x1, x2, x3), dim=1)

        x = F.relu(self.fc1(concatenated_output))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return self.fc4(x)


class DRL_Agent(object):
    """
    Q-Value Estimator neural network.

    This Q-Network is used for both the Action Network and the Target Network.
    """

    def __init__(self, PDP_size, PAS_size, CSL_size, Net_action_size, store_cache_length, learning_step_cur):
        # These are hyperparameter for the Policy learning
        self.gamma = 0.6  # discount rate
        self.learning_rate_init = 1e-3
        self.cur_learning_rate = self.learning_rate_init
        self.batch_size = 128
        self.tau = 0.0005
        self.device = torch.device("mps")
        self.initial_epsilon = 1 - 1e-4
        self.decay = (1 - 1e-4)*0.5
        self.min_epslion = 0.01

        # get size of state and action
        self.PDP_size = PDP_size  # int
        self.PAS_size = PAS_size  # int
        self.CSL_size = CSL_size  # int
        self.action_size = Net_action_size
        self.cache_length = store_cache_length  # int
        self.max_memory = 100000
        self.memory = deque(maxlen=self.max_memory)
        self.memory_len = len(self.memory)
        # 双向数组，超过长度删除另一侧的数据

        # LOSS_record Training Performance
        self.max_LOSS_record_len = self.max_memory
        self.LOSS_record = deque(maxlen=self.max_LOSS_record_len)
        self.LOSS_record_len = len(self.LOSS_record)

        # Reword record training performance
        self.max_reward_record_len = self.max_memory
        self.reward_record = deque(maxlen=self.max_LOSS_record_len)
        self.reward_record_len = len(self.LOSS_record)

        # create action and target DQN
        self.model = None
        # Net(self.PDP_size, self.PAS_size, self.CSL_size, self.action_size).to(self.device)
        self.target_model = None
        # Net(self.PDP_size, self.PAS_size, self.CSL_size, self.action_size).to(self.device)

        # Learning Part
        self.current_learning_step = learning_step_cur

        # Performance Statistics
        # lists for the states, actions and rewards

    def save_LOSS_record(self, loss_record_file_path):
        # save self.LOSS_record into file
        with open(loss_record_file_path, 'wb+') as f:
            pkl.dump(self.LOSS_record, f)
            f.close()
        return len(self.LOSS_record)

    def load_LOSS_record(self, loss_record_file_path):
        # save LOSS_record into file
        with open(loss_record_file_path, 'rb') as f:
            loss_record_temp = pkl.load(f)
            f.close()
        self.LOSS_record.extend(loss_record_temp)
        self.LOSS_record_len = len(self.LOSS_record)

    def add2LOSS_record(self, episode, loss):
        # add sample sequence to the LOSS_record
        self.LOSS_record.append((episode, loss))
        self.LOSS_record_len = len(self.LOSS_record)

    def save_reward_record(self, reward_record_file_path):
        # save self.reward_record into file
        with open(reward_record_file_path, 'wb+') as f:
            pkl.dump(self.reward_record, f)
            f.close()
        return len(self.reward_record)

    def load_reward_record(self, reward_record_file_path):
        # save reward_record into file
        with open(reward_record_file_path, 'rb') as f:
            reward_record_temp = pkl.load(f)
            f.close()
        self.reward_record.extend(reward_record_temp)
        self.reward_record_len = len(self.reward_record)

    def add2reward_record(self, epoch_idx, current_reward):
        # add sample sequence to the LOSS_record
        self.reward_record.append((epoch_idx, current_reward))
        self.reward_record_len = len(self.reward_record)

    def save_memory(self, memory_file_path):
        # save memory into file
        # print("Now we save memory file")
        with open(memory_file_path, 'wb+') as f:
            pkl.dump(self.memory, f)
            f.close()
        return len(self.memory)

    def load_memory(self, memory_file_path):
        # save memory into file
        # print("Now we load memory file")
        with open(memory_file_path, 'rb') as f:
            memory_temp = pkl.load(f)
            f.close()
        self.memory.extend(memory_temp)
        self.memory_len = len(self.memory)

    def add2replay_memory(self, PDP_state_seq, PAS_state_seq, CSL_state_seq, action_seq, reward_seq, next_PDP_state_seq,
                          next_PAS_state_seq, next_CSL_state_seq):
        if self.cache_length == 1:
            # add sample to the replay buffer
            self.memory.append(
                (PDP_state_seq, PAS_state_seq, CSL_state_seq, int(action_seq), reward_seq, next_PDP_state_seq,
                 next_PAS_state_seq, next_CSL_state_seq))
        else:
            for i in range(self.cache_length):
                # add sample sequence to the replay buffer
                self.memory.append(
                    (PDP_state_seq[i], PAS_state_seq[i], CSL_state_seq[i], int(action_seq[i]), reward_seq[i],
                     next_PDP_state_seq[i], next_PAS_state_seq[i], next_CSL_state_seq[i]))
        self.memory_len = len(self.memory)

    def get_action(self, PDP_state, PAS_state, CSL_state, epsilon_=0.05):
        # action calculated based epsilon-greedy policy
        # state = 模型输入的维度为(n_samples, input_dim)
        # n_samples = 1
        x1 = torch.unsqueeze(torch.FloatTensor(PDP_state), 0)
        x2 = torch.unsqueeze(torch.FloatTensor(PAS_state), 0)
        x3 = torch.unsqueeze(torch.FloatTensor(CSL_state), 0)
        """
        Forward Tensor to Device
        """
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        x3 = x3.to(self.device)
        self.model.eval()
        with torch.no_grad():
            actions_value = self.model.forward(x1, x2, x3)  # 通过对评估网络输入状态x，前向传播获得动作值`

        if np.random.random() < epsilon_:
            print('Random chosen of action')
            return np.random.choice(self.action_size) + 1, actions_value.data.cpu().numpy()
        else:
            print('Choose action based on DQN')
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
            return action + 1, actions_value.data.cpu().numpy()

    def train_DQN(self):
        """
        Learning
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate_init)
        loss_func = nn.SmoothL1Loss().to(self.device)  # Joint L1 and L2 loss function
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=5, eta_min=4e-5)
        """
        Train on a single minibatch
        """

        num_samples = min(self.batch_size, len(self.memory))
        replay_samples = random.sample(self.memory, num_samples)

        update_input_PDP = np.zeros((num_samples, self.PDP_size))
        update_input_PAS = np.zeros((num_samples, self.PAS_size))
        update_input_CSL = np.zeros((num_samples, self.CSL_size))

        next_input_PDP = np.zeros((num_samples, self.PDP_size))
        next_input_PAS = np.zeros((num_samples, self.PAS_size))
        next_input_CSL = np.zeros((num_samples, self.CSL_size))

        action = np.zeros((num_samples, 1))
        reward = np.zeros((num_samples, 1))
        # Shape of action/reward size is batch_size * 1

        if num_samples == 1:
            update_input_PDP[0, :] = replay_samples[-1][0]
            update_input_PAS[0, :] = replay_samples[-1][1]
            update_input_CSL[0, :] = replay_samples[-1][2]

            action[0, :] = replay_samples[-1][3] - 1
            reward[0, :] = replay_samples[-1][4]

            next_input_PDP[0, :] = replay_samples[-1][5]
            next_input_PAS[0, :] = replay_samples[-1][6]
            next_input_CSL[0, :] = replay_samples[-1][7]
        else:
            for i in range(num_samples):
                update_input_PDP[i, :] = replay_samples[i][0]
                update_input_PAS[i, :] = replay_samples[i][1]
                update_input_CSL[i, :] = replay_samples[i][2]

                action[i, :] = replay_samples[i][3] - 1
                reward[i, :] = replay_samples[i][4]

                next_input_PDP[i, :] = replay_samples[i][5]
                next_input_PAS[i, :] = replay_samples[i][6]
                next_input_CSL[i, :] = replay_samples[i][7]

        #  Q_value related to next observe state is estimated by the action DQN
        # Shape batch_size * Num_action
        PDP_state_th = torch.FloatTensor(update_input_PDP)
        PAS_state_th = torch.FloatTensor(update_input_PAS)
        CSL_state_th = torch.FloatTensor(update_input_CSL)

        action_th = torch.LongTensor(action.astype(int))
        reward_th = torch.FloatTensor(reward)

        next_PDP_state_th = torch.FloatTensor(next_input_PDP)
        next_PAS_state_th = torch.FloatTensor(next_input_PAS)
        next_CSL_state_th = torch.FloatTensor(next_input_CSL)

        """
        Forward Tensor to Device
        """
        PDP_state_th = PDP_state_th.to(self.device)
        PAS_state_th = PAS_state_th.to(self.device)
        CSL_state_th = CSL_state_th.to(self.device)

        action_th = action_th.to(self.device)
        reward_th = reward_th.to(self.device)

        next_PDP_state_th = next_PDP_state_th.to(self.device)
        next_PAS_state_th = next_PAS_state_th.to(self.device)
        next_CSL_state_th = next_CSL_state_th.to(self.device)

        """
        Training: Ref: https://zhuanlan.zhihu.com/p/260703124
        """

        q_val = self.model(PDP_state_th, PAS_state_th, CSL_state_th).gather(1, action_th)
        # Select action for successive state based on the model
        next_state_action_values = self.model(next_PDP_state_th, next_PAS_state_th, next_CSL_state_th).detach()
        # size = [batch_size, num_of_actions]
        _, next_action_batch = next_state_action_values.max(1)  # [batch_size]
        next_action_batch = next_action_batch.unsqueeze(1)  # [batch_size, 1]
        with torch.no_grad():
            q_next = self.target_model(next_PDP_state_th, next_PAS_state_th, next_CSL_state_th).gather(1, next_action_batch)
            # [batch_size, 1]
        q_val_train = reward_th + self.gamma * q_next

        """
        Gradient Descent
        """
        self.model.train()
        with torch.enable_grad():
            optimizer.zero_grad()
            loss = loss_func(q_val, q_val_train)
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
            # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
            scheduler.step(self.current_learning_step)
            optimizer.step()
            self.cur_learning_rate = optimizer.param_groups[-1]['lr']
            print('Learning step is %d，Current learning rate is %g, MSE Loss is %g'
                  % (self.current_learning_step, self.cur_learning_rate, loss.item()))

        return loss.item()

    def update_target_DQN(self):
        # Update the frozen target models: LINK: https://github.com/XinJingHao/DQN-DDQN-Pytorch/blob/main/DQN.py
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # load the saved action model
    def load_model(self, model_save_path):
        if os.path.exists(model_save_path):
            print('Load model')
            loaded_paras = torch.load(model_save_path, map_location=self.device)
            self.model.load_state_dict(loaded_paras)
        else:
            print("Error: No saved model of {}".format(model_save_path))

    # save the action model which is under training
    def save_model(self, model_save_path):
        model_state = copy.deepcopy(self.model.state_dict())
        torch.save(model_state, model_save_path)

    # load the saved target model
    def load_target_model(self, target_model_save_path):
        if os.path.exists(target_model_save_path):
            loaded_paras = torch.load(target_model_save_path, map_location=self.device)
            self.target_model.load_state_dict(loaded_paras)

    # save the target model which is under training
    def save_target_model(self, target_model_save_path):
        model_state = copy.deepcopy(self.target_model.state_dict())
        torch.save(model_state, target_model_save_path)



"""
The STA part
"""


def input_from_matlab(episode_, cache_length_, PDP_state_size_, PAS_state_size_, CSL_state_size_, action_size_,
                      root_path_, operation_mode_,
                      PDP_state_seq_, PAS_state_seq_, CSL_state_seq_,
                      action_seq_, reward_seq_,
                      next_PDP_state_seq_, next_PAS_state_seq_, next_CSL_state_seq_):
    data = dict()
    data['episode'] = int(episode_)
    data['cache_length'] = int(cache_length_)

    data['PDP_state_size'] = int(PDP_state_size_)
    data['PAS_state_size'] = int(PAS_state_size_)
    data['CSL_state_size'] = int(CSL_state_size_)

    data['action_size'] = int(action_size_)
    data['root_path'] = root_path_
    data['operation_mode'] = int(operation_mode_)

    data['PDP_state_seq'] = PDP_state_seq_.tolist()
    data['PAS_state_seq'] = PAS_state_seq_.tolist()
    data['CSL_state_seq'] = CSL_state_seq_.tolist()
    # array.tolist()：将数组转换为具有相同元素的列表（list）

    data['action_seq'] = action_seq_.tolist()
    data['reward_seq'] = reward_seq_.tolist()

    data['next_PDP_state_seq'] = next_PDP_state_seq_.tolist()
    data['next_PAS_state_seq'] = next_PAS_state_seq_.tolist()
    data['next_CSL_state_seq'] = next_CSL_state_seq_.tolist()
    # array.tolist()：将数组转换为具有相同元素的列表（list）

    return data


def mkdir(path):
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)
    # https://blog.csdn.net/zdc1305/article/details/106138491


if __name__ == "__main__":
    """
    import data from MATLAB part
    """
    input_data = input_from_matlab(episode_from_matlab, cache_length_from_matlab,
                                   PDP_state_size_from_matlab, PAS_state_size_from_matlab, CSL_state_size_from_matlab,
                                   action_size_from_matlab, Root_Path_File_from_matlab, Operation_mode_from_matlab,
                                   PDP_state_seq_from_matlab, PAS_state_seq_from_matlab, CSL_state_seq_from_matlab,
                                   action_seq_from_matlab, reward_seq_from_matlab,
                                   next_PDP_state_seq_from_matlab, next_PAS_state_seq_from_matlab,
                                   next_CSL_state_seq_from_matlab)

    DNN_model_save_dir = r"{}/DQN_models".format(input_data['root_path'])
    mkdir(DNN_model_save_dir)
    Action_DNN_model_save_path = os.path.join(DNN_model_save_dir, 'DQN_STA_model.pt')
    DNN_memory_file_path = os.path.join(DNN_model_save_dir, 'replay_memory_STA.pkl')

    if input_data['operation_mode'] == 0:  # Training mode
        agent = DRL_Agent(input_data['PDP_state_size'], input_data['PAS_state_size'], input_data['CSL_state_size'],
                          input_data['action_size'], input_data['cache_length'], input_data['episode'])
        agent.model = Net_STA(agent.PDP_size, agent.PAS_size, agent.CSL_size, agent.action_size).to(agent.device)
        agent.target_model = Net_STA(agent.PDP_size, agent.PAS_size, agent.CSL_size, agent.action_size).to(agent.device)
        """
            parameters
        """
        # Root File Path of simulation file
        # Samples_Path.txt文件路径
        target_DNN_model_save_path = os.path.join(DNN_model_save_dir, 'target_DQN_STA_model.pt')
        Loss_record_file_path = os.path.join(DNN_model_save_dir, 'Loss_record_STA.pkl')
        Reward_record_file_path = os.path.join(DNN_model_save_dir, 'Reward_record_STA.pkl')
        train_loss_path = os.path.join(DNN_model_save_dir, 'DRL_STA_train_loss.pkl')

        """
        Load Neural network
        """

        agent.add2replay_memory(input_data['PDP_state_seq'], input_data['PAS_state_seq'], input_data['CSL_state_seq'],
                                input_data['action_seq'], input_data['reward_seq'],
                                input_data['next_PDP_state_seq'], input_data['next_PAS_state_seq'],
                                input_data['next_CSL_state_seq'])

        if os.path.exists(Reward_record_file_path):
            agent.load_reward_record(Reward_record_file_path)

        if agent.cache_length == 1:
            sample_idx = (input_data['episode'] - 1) * agent.cache_length + 1
            agent.add2reward_record(sample_idx, input_data['reward_seq'])
        else:
            for i in range(agent.cache_length):
                sample_idx = (input_data['episode'] - 1) * agent.cache_length + 1 + i
                agent.add2reward_record(sample_idx, input_data['reward_seq'][i])

        reward_record_length = agent.save_reward_record(Reward_record_file_path)

        """
        train neural network
        """
        if agent.memory_len >= agent.batch_size:
            print('load saved model of STA in previous training')
            if os.path.exists(Action_DNN_model_save_path):
                agent.load_model(Action_DNN_model_save_path)

            if os.path.exists(target_DNN_model_save_path):
                agent.load_target_model(target_DNN_model_save_path)

            print('Training DQN')
            LOSS = agent.train_DQN()
            # save action DNN model
            print('save action DNN model of STA')
            agent.save_model(Action_DNN_model_save_path)
            """
                Record data for Tensorboard
            """
            if os.path.exists(Loss_record_file_path):
                agent.load_LOSS_record(Loss_record_file_path)
            agent.add2LOSS_record(agent.current_learning_step, LOSS)
            agent.save_LOSS_record(Loss_record_file_path)
            """
                update target DNN model
            """
            if agent.current_learning_step % 400 == 0:
                agent.tau = 1.0
                agent.update_target_DQN()
                # save target DNN model
                agent.save_target_model(target_DNN_model_save_path)
        else:
            LOSS = -1
            agent.current_learning_step = 0
            agent.cur_learning_rate = 0

        """
        output for matlab
        """
        output_for_MATLAB = dict()
        output_for_MATLAB['loss'] = LOSS
        output_for_MATLAB['reward_record_length'] = reward_record_length
        output_for_MATLAB['current_learning_step'] = agent.current_learning_step
        output_for_MATLAB['current_learning_rate'] = agent.cur_learning_rate

    else:  # In action mode
        print('In action mode of STA')
        agent = DRL_Agent(input_data['PDP_state_size'], input_data['PAS_state_size'], input_data['CSL_state_size'],
                          input_data['action_size'], input_data['cache_length'], input_data['episode'])
        agent.model = Net_STA(agent.PDP_size, agent.PAS_size, agent.CSL_size, agent.action_size).to(agent.device)
        if os.path.exists(Action_DNN_model_save_path):
            print('load STA DNN model')
            agent.load_model(Action_DNN_model_save_path)
        else:
            print('no STA DNN model')

        epsilon = max(agent.min_epslion, agent.initial_epsilon * (np.power(agent.decay, input_data['episode'] - 1)))
        action_choice, state_action_values = agent.get_action(input_data['PDP_state_seq'],
                                                              input_data['PAS_state_seq'],
                                                              input_data['CSL_state_seq'], epsilon)

        output_for_MATLAB = dict()
        output_for_MATLAB['action_idx'] = action_choice
        output_for_MATLAB['state_action_values'] = state_action_values
