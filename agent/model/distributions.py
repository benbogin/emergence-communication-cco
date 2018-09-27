import torch.nn as nn
import torch.nn.functional as F


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x, softmax=False, temperature=1):
        x = self.linear(x) / temperature
        if softmax:
            return F.softmax(x, dim=-1)
        return x

    def sample(self, x, deterministic, temperature=1):
        probs = self(x, softmax=True, temperature=temperature)

        # print(probs.data.numpy())
        if deterministic is False:
            action = probs.multinomial()
        else:
            action = probs.max(1)[1].unsqueeze(1)
        return action, probs

    def evaluate_actions(self, x, actions):
        x = self(x)

        log_probs = F.log_softmax(x, dim=-1)
        probs = F.softmax(x, dim=-1)

        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        dist_entropy = -(log_probs * probs).sum(-1)
        return action_log_probs, dist_entropy
