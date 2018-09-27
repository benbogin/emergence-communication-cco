import torch


def get_sentence_lengths(comm_input):
    return torch.sum((comm_input+1).sign(), dim=1)


def get_relevant_state(states, sentence_lengths):
    return torch.gather(states, dim=1, index=(sentence_lengths -1).view(-1, 1, 1).expand(states.size(0), 1, states.size(2)))


def mask_out_padding_tokens(actions):
    return (actions + 1).clamp(min=0).sign()


def array_to_str_sentence(sentence, padding=-1):
    return ','.join([str(d) for d in sentence if d != padding])


def str_sentence_to_array(sentence):
    return sentence.split(',')

