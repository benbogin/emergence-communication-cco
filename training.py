import copy
import os

from config import args
from environment.make_envs import make_env
from environment.multi_vec_env import MultiVecEnv
from environment.tasks.task_manager import TaskManager


def pretrain(agents, experiment_path):
    return train(None, agents, experiment_path)


def train(speaker_agents, actor_agents, experiment_path):
    num_steps, total_steps, num_processes = args.num_steps, args.total_steps, args.num_processes
    num_updates = total_steps // num_steps // num_processes

    train_task_manager = TaskManager(args.n_missions)
    test_task_manager = TaskManager(args.n_missions, total_tasks=100, test_mode=True)

    train_envs = MultiVecEnv([make_env('GridWorldEnv-v0') for i in range(num_processes)], train_task_manager)
    test_envs = MultiVecEnv([make_env('GridWorldEnv-v0') for i in range(num_processes)], test_task_manager)

    states, inventories = train_envs.reset()

    for j in range(num_updates):
        if j % 10 == 0:
            evaluate_on_test_set(experiment_path, speaker_agents, actor_agents, test_envs, num_steps)

        states, inventories = run_env_steps(speaker_agents, actor_agents, states, inventories, train_envs, num_steps)

        if j % 75 == 0 and j > 0:
            if speaker_agents is not None:
                speaker_agents.save_model(j * num_steps * num_processes, os.path.join(experiment_path, 'checkpoints'))
            actor_agents.save_model(j * num_steps * num_processes, os.path.join(experiment_path, 'checkpoints'))

    print("Finished")

    train_envs.close()
    test_envs.close()

    return experiment_path


def run_env_steps(speaker_agents, actor_agents, states, inventories, envs, num_steps, learn=True, deterministic=False):
    speaker_steps = []
    actor_steps = []

    actor_agents.new_round()
    if speaker_agents is not None:
        speaker_agents.new_round()

    for step in range(num_steps):
        comm_actions, comm_probs = None, None
        if speaker_agents is not None:
            speaker_world_rep, speaker_value, comm_actions, comm_probs = speaker_agents.select_comm_action(
                states, inventories, deterministic)

        actor_world_rep, actor_value, env_actions, env_action_probs = actor_agents.select_env_action(
            states, inventories, comm_actions, deterministic)

        states, inventories, reward, env_done, info = envs.step(env_actions)

        actor_steps.append((states, inventories, actor_world_rep, env_actions, comm_actions, actor_value, reward,
                            env_done))

        actor_agents.step_ended(step, env_done)

        if speaker_agents is not None:
            speaker_agents.step_ended(step, env_done)
            speaker_steps.append((states, inventories, speaker_world_rep, comm_actions, None, speaker_value, reward,
                                  env_done))

    if learn:
        actor_agents.store(actor_steps)
        actor_agents.learn()

        if speaker_agents is not None:
            speaker_agents.store(speaker_steps)
            speaker_agents.learn()

    return states, inventories


def evaluate_on_test_set(experiment_path, speaker_agents, actor_agents, test_envs, num_steps):
    eval_states, eval_inventories = test_envs.reset()
    test_envs.task_manager.reset_report()
    test_speaker_agents = copy.copy(speaker_agents)
    if speaker_agents is not None:
        test_speaker_agents.reset_state()
    test_actor_agents = copy.copy(actor_agents)
    test_actor_agents.reset_state()
    while not test_envs.task_manager.report_full():
        eval_states, eval_inventories = run_env_steps(test_speaker_agents, test_actor_agents, eval_states, eval_inventories,
                                                      test_envs, num_steps, learn=False, deterministic=True)
    success_percentage = test_envs.task_manager.get_success_percentage()

    print("evaluate_on_test_set:", experiment_path, success_percentage,
          "unsuccessful tasks:", test_envs.task_manager.get_unsuccessful_tasks())

    return success_percentage

