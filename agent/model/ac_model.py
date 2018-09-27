import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter

from agent.agent_type import AgentType
from config import args
from utils.seq import get_sentence_lengths, mask_out_padding_tokens, get_relevant_state
from utils.torch_functions import cuda
import numpy as np
from agent.model.distributions import Categorical
from utils.torch_functions import get_fts


class ActorCriticModel(nn.Module):
    def __init__(self, input_shape, inventory_shape, action_space, vocab_size, sen_len, agent_type, acting):
        super(ActorCriticModel, self).__init__()

        self.type = agent_type
        self.acting = acting
        self.action_space = action_space
        self.vocab_size = vocab_size

        self.sen_len = sen_len

        self.rep_dim = 50

        input_shape = (input_shape[2], input_shape[0], input_shape[1])
        inventory_shape = (inventory_shape[2], inventory_shape[0], inventory_shape[1])

        self.obs_filters = 6

        self.conv_obs = nn.Sequential(
            nn.Conv2d(input_shape[0], self.obs_filters, kernel_size=3, stride=3, padding=0),
            nn.Tanh(),
        )

        conv_obs_out_size = get_fts(input_shape, self.conv_obs)
        conv_inv_out_size = get_fts(inventory_shape, self.conv_obs)
        conv_inv_out_size_flat = get_fts(inventory_shape, self.conv_obs, flat=True)

        if args.model == 'gru':
            self.hobs_linear = nn.Sequential(
                nn.Linear(conv_inv_out_size_flat, conv_inv_out_size_flat),
                nn.Tanh()
            )
        elif args.model == 'bow':
            self.hobs_linear = nn.Sequential(
                nn.Linear(self.rep_dim, conv_inv_out_size_flat),
                nn.Tanh()
            )

        obs2_filters = 8
        self.conv_obs2 = nn.Sequential(
            nn.Conv2d(conv_inv_out_size[1]*conv_inv_out_size[2], obs2_filters, kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
        )
        conv_obs2_in_size = (conv_inv_out_size[1]*conv_inv_out_size[2], conv_obs_out_size[1], conv_obs_out_size[2])
        conv_obs2_out_size = get_fts(conv_obs2_in_size, self.conv_obs2, flat=True)

        self.world_linear = nn.Sequential(
            nn.Linear(conv_obs2_out_size + (conv_inv_out_size[1] * conv_inv_out_size[2])**2, self.rep_dim),
            nn.Tanh()
        )

        embeddings_vocab_size = vocab_size
        if args.algorithm == 'pg':
            embeddings_vocab_size += 1  # add for SOS token
        self.listener_embeddings = nn.Embedding(embeddings_vocab_size, self.rep_dim, padding_idx=-1)

        self.linear_critic = nn.Linear(self.rep_dim, 1)

        self.actor_lstm = nn.GRU(self.rep_dim, self.rep_dim, num_layers=1)
        self.actor_comm_lstm = nn.GRU(self.rep_dim, conv_inv_out_size_flat, num_layers=1, batch_first=True)
        self.speaker_lstm = nn.GRU(self.rep_dim, self.rep_dim, num_layers=1)

        self.action_dist = Categorical(self.rep_dim, action_space)

        if args.algorithm == 'pg':
            self.comm_dist = Categorical(self.rep_dim, vocab_size+1)  # add 1 for EOS token

        self.empty_symbol_embedding = Parameter(torch.zeros((1, 1, self.rep_dim)), requires_grad=True)

        if args.algorithm == 'cco' and self.type == AgentType.speaker and not acting:
            self.freeze_all_except([
                self.listener_embeddings,
                self.actor_comm_lstm,
                self.hobs_linear
            ])

    def freeze_all_except(self, layers):
        for m in self.modules():
            m.requires_grad = False
            if hasattr(m, 'weight'):
                m.weight.requires_grad = False
        for l in layers:
            for m in l.modules():
                m.requires_grad = True
                if hasattr(m, 'weight'):
                    m.weight.requires_grad = True

    def init_params(self):
        nn.init.xavier_normal(self.action_dist.linear.weight)

    def repackage_hidden(self, batch_size, prev_state):
        if prev_state is not None:
            hidden = prev_state.data
        else:
            hidden = torch.zeros((1, batch_size, self.rep_dim))
        return cuda(Variable(hidden))

    def mask_hidden_states(self, prev_state, mask):
        mask = cuda(Variable(torch.unsqueeze(torch.unsqueeze(1 - torch.from_numpy(mask.astype(np.float)), 0), 2).float()))
        return mask * prev_state

    def get_rep_dim(self):
        """Return dimension of state representation"""
        return self.rep_dim

    def forward(self, world_state, inventory_state, comm_state, lstm_hidden_state, full_sentence=True, deterministic=False, hide_goal=False, force_goal=False):
        """
        world_state: batch x 2 (state and goal) x width x height x 3 (colors)
        """

        batch_size = world_state.data.shape[0]

        if self.acting:
            world_state = world_state[:, 1]  # listener's world
        else:
            world_state = world_state[:, 0]  # speaker's world

        # batch x 2 (state and goal) x 3 (colors) width x height
        world_state = torch.transpose(world_state, 1, 3).contiguous()
        inventory_state = torch.transpose(inventory_state, 2, 4).contiguous()

        obs_state = world_state
        inv_state = inventory_state[:, 0]  # current inventory_state of agent
        inv_goal_state = inventory_state[:, 1]  # goal inventory_state

        obs_state = self.conv_obs(obs_state)
        inv_state = self.conv_obs(inv_state)

        conv_inv_out_size = inv_state.size()
        conv_obs_out_size = obs_state.size()

        if comm_state is not None:
            sentence_lengths = get_sentence_lengths(comm_state)

            if args.algorithm == 'pg' and args.model == 'gru':
                SOS_symbol = self.vocab_size
                comm_state = comm_state.clone()
                comm_state[comm_state == self.vocab_size] = -1  # remove EOS symbol
                sentence_lengths = get_sentence_lengths(comm_state)
                comm_state = torch.cat((cuda(Variable(torch.ones(batch_size, 1).long())) * SOS_symbol, comm_state), dim=1)
                sentence_lengths += 1

            comm_input = self.listener_embeddings(comm_state.clamp(0))  # clamp to avoid error for -1 padding

            if args.model == 'gru':
                goal_state, _ = self.actor_comm_lstm(comm_input, None)
                goal_state = get_relevant_state(goal_state, sentence_lengths).squeeze(1)
                goal_state = self.hobs_linear(goal_state).view(-1, int(np.prod(conv_inv_out_size[1:])))
            elif args.model == 'bow':
                mask = mask_out_padding_tokens(comm_state).unsqueeze(2)
                bag_of_embeddings = (comm_input * mask.float()).sum(dim=1) / sentence_lengths.unsqueeze(0).unsqueeze(2).float()
                goal_state = self.hobs_linear(bag_of_embeddings).view(-1, int(np.prod(conv_inv_out_size[1:])))
        else:
            goal_state = self.conv_obs(inv_goal_state)

        obs_state = obs_state.view(batch_size, self.obs_filters, -1).transpose(1,2) / (obs_state.view(batch_size, self.obs_filters, -1).transpose(1,2).norm(dim=-1)).unsqueeze(-1)
        goal_state = goal_state.view(batch_size, self.obs_filters, -1).transpose(1,2) / (goal_state.view(batch_size, self.obs_filters, -1).transpose(1,2).norm(dim=-1)).unsqueeze(-1)
        inv_state = inv_state.view(batch_size, self.obs_filters, -1).transpose(1,2) / (inv_state.view(batch_size, self.obs_filters, -1).transpose(1,2).norm(dim=-1)).unsqueeze(-1)
        obs_state = torch.matmul(obs_state, goal_state.transpose(1,2))
        inv_state = torch.matmul(inv_state, goal_state.transpose(1,2))

        obs_state = self.conv_obs2(obs_state.transpose(1,2).contiguous().view(batch_size, conv_inv_out_size[2]*conv_inv_out_size[3], conv_obs_out_size[2], conv_obs_out_size[3])).view(batch_size, -1)
        hidden_state = self.world_linear(torch.cat((obs_state, inv_state.view(batch_size, -1)), dim=-1))

        comm_lstm_hidden_state = hidden_state.unsqueeze(0)
        symbol_embedding = self.empty_symbol_embedding.repeat(1, batch_size, 1)

        hidden_state, comm_lstm_hidden_state = self.speaker_lstm(symbol_embedding, comm_lstm_hidden_state)

        if args.algorithm == 'cco':
            hidden_state = hidden_state.squeeze(0)
            actions, actions_probs = self.action_dist.sample(hidden_state, deterministic)
        else:
            if self.acting or not full_sentence:
                actions, actions_probs = self.action_dist.sample(hidden_state.squeeze(0), deterministic)
            else:
                EOS_symbol = self.vocab_size
                finished_procs = [0 for _ in range(batch_size)]

                actions, actions_probs = [], []
                hidden_states = []

                for i in range(self.sen_len):
                    action, probs = self.comm_dist.sample(hidden_state.squeeze(0), deterministic, args.pg_speaker_softmax_temp)
                    symbol_embedding = self.listener_embeddings(action.squeeze(1).clamp(min=0)).unsqueeze(0)

                    action = action.data
                    for p, a in enumerate(action):
                        if a[0] == EOS_symbol:
                            finished_procs[p] = 1
                        if finished_procs[p]:
                            action[p] = EOS_symbol

                    actions.append(action)
                    actions_probs.append(probs)
                    hidden_states.append(hidden_state)

                    hidden_state, comm_lstm_hidden_state = self.speaker_lstm(symbol_embedding, comm_lstm_hidden_state)

                actions = torch.cat(actions, dim=1)
                actions_probs = torch.stack(actions_probs, dim=1)
                hidden_state = torch.cat(hidden_states, dim=0)
                hidden_state = hidden_state.transpose(0, 1).contiguous()

        return self.linear_critic(hidden_state), hidden_state, lstm_hidden_state, actions, actions_probs

    def evaluate_actions(self, world_rep, actions, first_env_step, eos_mask=None):
        if args.algorithm == 'cco':
            assert self.acting
            action_log_probs, action_entropy = self.action_dist.evaluate_actions(world_rep, actions)

            action_entropy = action_entropy.mean()

            return action_log_probs, action_entropy
        elif args.algorithm == 'pg':
            if self.acting:
                action_log_probs, action_entropy = self.action_dist.evaluate_actions(world_rep, actions)
            else:
                action_log_probs, action_entropy = self.comm_dist.evaluate_actions(world_rep, actions)

            if not self.acting:
                first_env_step = cuda(
                    Variable(torch.from_numpy(first_env_step.repeat(self.sen_len).astype(np.float32)).view(-1),
                             requires_grad=False))
                eos_mask = Variable(1-eos_mask).float()
                action_entropy = action_entropy * first_env_step * eos_mask
                action_log_probs = action_log_probs * first_env_step * eos_mask

            action_entropy = action_entropy.mean()

            return action_log_probs, action_entropy
