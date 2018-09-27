import torch
from torch.autograd import Variable
import numpy as np
from torch.nn.functional import softmax


def one_hot(batch, depth, first_is_none=False):
    flat = batch.view(-1, 1)
    encoded = cuda(torch.FloatTensor(flat.shape[0], depth)).zero_().scatter_(1, flat, 1.0)
    if first_is_none:
        encoded[:, 0] = 0
        encoded = encoded[:, 1:].contiguous()
        return Variable(encoded.view(*batch.shape, depth-1))
    else:
        return Variable(encoded.view(*batch.shape, depth))


def get_fts(in_size, fts, flat=False):
    """Calculate size of flattened features given a function and input size"""
    f = fts(Variable(torch.ones(1,*in_size)))
    if flat:
        return int(np.prod(f.size()[1:]))
    else:
        return list(f.size()[1:])


cuda_available = torch.cuda.is_available()


def cuda(tensor):
    if cuda_available:
        return tensor.cuda()
    else:
        return tensor


def temperature_softmax(arr, temperature=1.0):
    # helper function to sample an index from a probability array
    a = arr - arr.max(dim=1)[0].unsqueeze(1)
    a = a / temperature
    return torch.exp(a) / torch.sum(torch.exp(a), dim=1).unsqueeze(1)