import gym
import torch
import os
import numpy as np

from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm

from agent.agent_type import AgentType
from agent.model.ac_model import ActorCriticModel
from agent.storage import RolloutStorage
from config import args
from utils.torch_functions import cuda, temperature_softmax


class VecAgent:
    def __init__(self, agent_type, acting):
        dummy_env = gym.make('GridWorldEnv-v0')  # just to extract observation/action space

        self.n_processes = args.num_processes
        self.vocab_size = args.vocab_size
        self.rollouts = RolloutStorage(args.num_steps, args.num_processes)
        self.action_space = dummy_env.action_space.n
        self.n_steps = args.num_steps
        self.sentence_len = args.sentence_len
        self.agent_type = agent_type
        self.acting = acting

        obs_shape, inventory_shape = dummy_env.observation_space

        self.model = cuda(ActorCriticModel(obs_shape, inventory_shape, self.action_space, self.vocab_size,
                                           self.sentence_len, agent_type, acting))

        self.value_loss_coef = args.value_loss_coef

        self.gamma = args.gamma
        self.global_step = 0

        params = [param for param in self.model.parameters() if param.requires_grad]
        if not acting:
            lr = args.lr_speaker
        else:
            lr = args.lr_listener
        self.optimizer = torch.optim.RMSprop(params, lr, eps=1e-5, alpha=0.99)

        self.lstm_hidden_state = None
        self.last_comm_actions = [None for _ in range(self.n_processes)]

        self.speaker_dist_loss = []
        self.used_sentences = []

    def select_comm_action(self, all_states, all_inventories, deterministic=False):
        if args.algorithm == 'cco':
            return self.select_comm_action_obverter(all_states, all_inventories, deterministic=deterministic)
        elif args.algorithm == 'pg':
            return self.select_comm_action_pg(all_states, all_inventories, deterministic=deterministic)

    def select_env_action(self, states, inventories, comm_actions, deterministic=False):
        value, hidden_states, lstm_hidden_state, actions, probs = self.model(
            Variable(cuda(torch.from_numpy(states)).float()),
            Variable(cuda(torch.from_numpy(inventories)).float()),
            None if comm_actions is None else Variable(cuda(torch.from_numpy(comm_actions)).long()),
            self.lstm_hidden_state,
            deterministic=deterministic,
            full_sentence=False
        )

        self.lstm_hidden_state = lstm_hidden_state

        hidden_states = hidden_states.view([self.n_processes, -1])
        value = value.view([self.n_processes])
        actions = actions.view([self.n_processes]).data.cpu().numpy()
        probs = probs.view([self.n_processes, -1]).data.cpu().numpy()

        return hidden_states, value, actions, probs

    def reset_state(self):
        self.lstm_hidden_state = None
        self.last_comm_actions = [None for _ in range(self.n_processes)]

    def new_round(self):
        self.lstm_hidden_state = self.model.repackage_hidden(self.n_processes, self.lstm_hidden_state)
        self.rollouts.new_round()
        self.speaker_dist_loss = []

    def step_ended(self, step, mask):
        self.rollouts.update_first_env_step(step, mask)

        self.lstm_hidden_state = self.model.mask_hidden_states(self.lstm_hidden_state, mask)
        self.last_comm_actions = [None if m else last_action for m, last_action in zip(mask, self.last_comm_actions)]

    def store(self, steps):
        state, inv, reps, actions, comm_input, value, reward, done = zip(*steps)
        masks = (1 - np.array(done)).astype(np.float32)
        self.rollouts.insert(
            torch.from_numpy(np.array(state)).float(),
            torch.from_numpy(np.array(inv)).float(),
            self.lstm_hidden_state,
            torch.stack(reps),
            torch.from_numpy(np.stack(actions)),
            None if comm_input[0] is None else torch.from_numpy(np.stack(comm_input)).long(),
            torch.stack(value),
            torch.from_numpy(np.array(reward).astype(np.float32)),
            torch.from_numpy(masks)
        )

    def learn(self):
        self.global_step += 1
        actions = self.rollouts.actions.long()
        comm_input = self.rollouts.comm_inputs

        next_value = cuda(self.model(
            cuda(Variable(self.rollouts.world_states[-1], volatile=True)),
            cuda(Variable(self.rollouts.inventories[-1], volatile=True)),
            None if comm_input is None else cuda(Variable(comm_input[-1], volatile=True)),
            self.rollouts.lstm_hidden_states,
            full_sentence=False
        )[0].data)

        self.rollouts.compute_returns(next_value, self.gamma)

        if args.algorithm == 'cco':
            total_loss = self.learn_obverter(actions)
        elif args.algorithm == 'pg':
            total_loss = self.learn_pg(actions)

        self.optimizer.zero_grad()
        total_loss.backward()

        clip_grad_norm(self.model.parameters(), 0.5)

        self.optimizer.step()

    def learn_obverter(self, actions):
        values = self.rollouts.value_preds.view(self.n_steps, self.n_processes)
        advantages = critic = Variable(self.rollouts.returns[:-1]) - values
        value_loss = advantages.pow(2).mean()
        total_loss = value_loss * self.value_loss_coef

        if self.agent_type == AgentType.speaker and not self.acting:
            if len(self.speaker_dist_loss) > 0:
                speaker_dist_loss = torch.stack(self.speaker_dist_loss)
                dist_loss = (speaker_dist_loss * Variable(critic.data.unsqueeze(1))).clamp(min=0).mean()
                dist_loss += (speaker_dist_loss * Variable(critic.data.unsqueeze(1))).clamp(max=0).mean() * \
                             args.failed_penalty
                total_loss += dist_loss
        else:
            action_log_probs, action_entropy = self.model.evaluate_actions(
                self.rollouts.reps.view(-1, self.model.get_rep_dim()),
                Variable(actions.contiguous().view(-1)),
                self.rollouts.first_env_step[:-1]
            )

            action_log_probs = action_log_probs.contiguous().view(self.n_steps, self.n_processes)
            action_loss = -(
                        Variable(critic.data) * action_log_probs).mean()  # critic is wrapped with variable to freeze it

            entropy_loss = action_entropy * args.entropy_coef_listener

            total_loss += action_loss - entropy_loss
        return total_loss

    def save_model(self, global_step, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        name = 'checkpoint-%d-%s' % (self.agent_type.value, global_step)
        torch.save(self.model.state_dict(), os.path.join(dir, "%s.ckp" % name))

    def load_checkpoint(self, path, skip_vocab_params=False):
        if torch.cuda.is_available():
            update_params = torch.load(path)
        else:
            update_params = torch.load(path, map_location=lambda storage, loc: storage)

        strict = True

        # skip_vocab_params flag allows training agents with vocabulary size (that forces a different network) that
        # is different from the one used in pretraining
        if skip_vocab_params:
            params = ['listener_embeddings.weight', 'comm_dist.linear.weight', 'comm_dist.linear.bias', 'symbol_embedding.weight',
                      'hobs_linear.0.weight', 'hobs_linear.0.bias']
            for param in params:
                if param in update_params:
                    del update_params[param]
            strict = False

        self.model.load_state_dict(update_params, strict=strict)

    def select_comm_action_obverter(self, all_states, all_inventories, deterministic):
        actions = np.array([[-1 for _ in range(self.sentence_len)] if a is None else a for a in self.last_comm_actions])

        value, target_hidden_states, _, _, _ = self.model(Variable(cuda(torch.from_numpy(all_states)).float()),
                                                          Variable(cuda(torch.from_numpy(all_inventories)).float()),
                                                          None, None, deterministic=deterministic)

        # procs are only relevant if it's the first step in the episode
        relevant_procs = [i for i, prev_action in enumerate(self.last_comm_actions) if prev_action is None]

        dist_loss = [Variable(cuda(torch.zeros(self.n_processes).float())) for _ in range(self.sentence_len)]

        if len(relevant_procs) > 0:
            for l in range(self.sentence_len):
                states = all_states[relevant_procs]
                inventories = all_inventories[relevant_procs]
                n_processes = len(states)

                next_symbol = np.repeat(np.expand_dims(np.arange(0, self.vocab_size), 0), n_processes, 0)

                if l > 0:
                    run_communications = np.concatenate((np.expand_dims(actions[relevant_procs, :l].transpose(),
                                                                        2).repeat(self.vocab_size, axis=2),
                                                         np.expand_dims(next_symbol, 0)), axis=0)
                else:
                    run_communications = np.expand_dims(next_symbol, 0)

                expanded_states = np.tile(states, (self.vocab_size, 1, 1, 1, 1))
                expanded_inventories = np.tile(inventories, (self.vocab_size, 1, 1, 1, 1))

                _, hidden_states, _, _, _ = self.model(Variable(cuda(torch.from_numpy(expanded_states)).float()),
                                                       Variable(cuda(torch.from_numpy(expanded_inventories)).float()),
                                                       Variable(cuda(torch.from_numpy(
                                                           run_communications.transpose().reshape(-1, 1 + l))).long()),
                                                       None)

                target_hidden_states = Variable(target_hidden_states.data)  # freeze gradients
                relevant_target_hidden_states = target_hidden_states[
                    cuda(torch.from_numpy(np.array(relevant_procs)).long())]
                all_distances = [h.dist(relevant_target_hidden_states[i % n_processes]) for i, h in
                                 enumerate(hidden_states)]
                all_distances = torch.cat(all_distances).view((self.vocab_size, n_processes))

                n_options = self.vocab_size
                if l > 0:
                    # add EOS token (-1) as a selectable option
                    all_distances = torch.cat((all_distances, dist_loss[l - 1][relevant_procs].unsqueeze(0)))
                    n_options += 1

                    run_communications = np.concatenate(
                        (run_communications, np.ones((l + 1, n_processes, 1), np.int32) * -1), axis=2)
                    run_communications[:-1, :, -1] = run_communications[:-1, :, 0]

                all_distances = all_distances.transpose(0, 1).contiguous()

                if deterministic or args.speaker_softmax_temp == 0:
                    sel_comm_idx = torch.min(all_distances.view((n_processes, -1)), dim=1)[1]
                else:
                    distances_distribution = temperature_softmax(-all_distances, temperature=args.speaker_softmax_temp)
                    sel_comm_idx = distances_distribution.multinomial(1).squeeze(1)

                sel_comm_idx_flat = (sel_comm_idx + Variable(cuda(torch.arange(0, n_processes) * n_options).long()))
                sel_distances = all_distances.view(-1)[sel_comm_idx_flat.data]
                timestep_distance_losses = [Variable(torch.zeros(1)) for _ in range(self.n_processes)]

                comm = run_communications[:, np.arange(len(relevant_procs)),
                       sel_comm_idx.data.cpu().numpy()].transpose()
                finished_p = []
                for i, (action, p) in enumerate(zip(comm, relevant_procs)):
                    if action[-1] == -1:
                        finished_p.append(p)
                    else:
                        timestep_distance_losses[p] = sel_distances[i]

                        dist_loss[l][p] = timestep_distance_losses[p]
                    for j, symb in enumerate(action):
                        actions[p][j] = symb

                for p in finished_p:
                    relevant_procs.remove(p)

                if len(relevant_procs) == 0:
                    break

        actions = np.array(actions)

        self.last_comm_actions = actions

        probs = np.zeros((self.n_processes, self.sentence_len, self.vocab_size))
        for p in range(self.n_processes):
            for pos in range(self.sentence_len):
                probs[p, pos, actions[p, pos]] = 1

        self.speaker_dist_loss.append(torch.stack(dist_loss))

        return target_hidden_states, value, actions, probs

    def select_comm_action_pg(self, all_states, all_inventories, deterministic):
        value, hidden_states, _, actions, probs = self.model(Variable(cuda(torch.from_numpy(all_states)).float()),
                                                                 Variable(cuda(torch.from_numpy(all_inventories)).float()),
                                                                 None, None, deterministic=deterministic)

        actions = actions.cpu().numpy()
        actions = np.array([new_action if prev_action is None else prev_action for new_action, prev_action in zip(actions, self.last_comm_actions)])
        self.last_comm_actions = actions

        probs = probs.data.cpu().numpy()

        return hidden_states, value, actions, probs

    def learn_pg(self, actions):
        values = self.rollouts.value_preds.view(self.n_steps, self.n_processes, -1)
        advantages = critic = Variable(self.rollouts.returns[:-1].unsqueeze(2)) - values
        value_loss = advantages.pow(2).mean()

        if not self.acting:
            eos_mask = ((torch.cat((cuda(torch.zeros(actions[:, :, 0:1].size())).long(), actions[:, :, :-1]), dim=2) == self.vocab_size) * (actions == self.vocab_size)).view(-1)
        else:
            eos_mask = None

        action_log_probs, action_entropy = self.model.evaluate_actions(
            self.rollouts.reps.view(-1, self.model.get_rep_dim()),
            Variable(actions.contiguous().view(-1)),
            self.rollouts.first_env_step[:-1],
            eos_mask
        )

        action_log_probs = action_log_probs.contiguous().view(self.n_steps, self.n_processes, -1)
        action_loss = -(Variable(critic.data) * action_log_probs).mean()  # critic is wrapped with variable to freeze it

        self.optimizer.zero_grad()

        entropy_loss = action_entropy * (args.entropy_coef_speaker if self.agent_type == AgentType.speaker else args.entropy_coef_listener)

        total_loss = value_loss * self.value_loss_coef + action_loss - entropy_loss

        return total_loss
