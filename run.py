import os
import time

from agent.agent_type import AgentType
from agent.vec_agent import VecAgent
from config import args
from training import pretrain, train

if __name__ == "__main__":
    train_type = args.train_type
    experiment_path = os.path.join('experiments', train_type, str(round(time.time()))[-6:])

    if train_type == 'pretrain_speaker' or train_type == 'pretrain_listener':
        agent_type = train_type.split('_')[1]
        agent = VecAgent(AgentType.from_name(agent_type), acting=True)
        pretrain(agent, experiment_path)
    elif train_type == 'train_joint':
        speaker_agent = VecAgent(AgentType.speaker, acting=False)
        listener_agent = VecAgent(AgentType.listener, acting=True)

        if args.restore_speaker is not None:
            speaker_agent.load_checkpoint(args.restore_speaker)
        if args.restore_listener != "None":
            listener_agent.load_checkpoint(args.restore_listener)

        train(speaker_agent, listener_agent, experiment_path)
    else:
        raise(Exception("No such training type: %s" % train_type))