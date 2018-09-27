import torch
from utils.torch_functions import cuda
import numpy as np


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes):
        self.num_steps = num_steps
        self.world_states = None
        self.inventories = None
        self.comm_input = None
        self.lstm_hidden_states = None
        self.rewards = None
        self.returns = cuda(torch.zeros(num_steps + 1, num_processes))
        self.actions = None
        self.comm_inputs = None
        self.masks = None
        self.first_env_step = np.ones((num_steps + 1, num_processes), np.int32)
        self.reps = None
        self.value_preds = None
        self.oracle_actions = None

    def insert(self, current_state, current_inv, lstm_hidden_states, reps, action, comm_inputs, value_pred, reward, mask):
        self.world_states = current_state
        self.inventories = current_inv
        self.lstm_hidden_states = lstm_hidden_states
        self.reps = cuda(reps)
        self.actions = cuda(action)
        self.comm_inputs = None if comm_inputs is None else cuda(comm_inputs)
        self.value_preds = cuda(value_pred)
        self.rewards = cuda(reward)
        self.masks = cuda(mask)

    def compute_returns(self, next_value, gamma):
        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.size(0))):
            self.returns[step] = self.returns[step + 1] * \
                gamma * self.masks[step] + self.rewards[step]

    def update_first_env_step(self, step, mask):
        self.first_env_step[step + 1] = np.array(mask).astype(np.int32)

    def new_round(self):
        self.first_env_step[0] = self.first_env_step[-1]
