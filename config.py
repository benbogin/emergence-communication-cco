import argparse

parser = argparse.ArgumentParser(description='Emergence of Communication in an Interactive World with Consistent Speakers')

parser.add_argument('train_type', choices=['pretrain_speaker', 'pretrain_listener', 'train_joint'])

parser.add_argument('--model', default='gru')
parser.add_argument('--algorithm', default='cco', choices=['cco', 'pg'])

parser.add_argument('--n-missions', type=int, default=1)
parser.add_argument('--n-colors', type=int, default=3)
parser.add_argument('--n-numbers', type=int, default=3)

parser.add_argument('--restore-speaker', default=None, help='speaker\'s path to checkpoint file')
parser.add_argument('--restore-listener', default=None, help='listener\'s path to checkpoint file')
parser.add_argument('--no-video-capture', action='store_true', help='do not capture & save videos and logs (debug)')

parser.add_argument('--lr-speaker', type=float, default=4e-4, help='speaker learning rate')
parser.add_argument('--lr-listener', type=float, default=1e-3, help='listener learning rate')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards')
parser.add_argument('--num-processes', type=int, default=10, help='how many training CPU processes to use')
parser.add_argument('--num-steps', type=int, default=5, help='number of forward steps in A2C')
parser.add_argument('--total-steps', type=int, default=10000000, help='total number of steps to train')
parser.add_argument('--value-loss-coef', type=float, default=0.5, help='value loss coefficient')
parser.add_argument('--entropy-coef-speaker', type=float, default=0.06, help='speaker entropy term coefficient')
parser.add_argument('--entropy-coef-listener', type=float, default=0.06, help='listener entropy term coefficient')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--cell-size', type=int, default=3)
parser.add_argument('--world-size', type=int, default=5)

parser.add_argument('--vocab-size', type=int, default=10, help='vocabulary size available to agents')
parser.add_argument('--sentence-len', type=int, default=3, help='number of words per sentence')
parser.add_argument('--skip-vocab-params', action='store_true')
parser.add_argument('--speaker-softmax-temp', type=float, default=0.0)
parser.add_argument('--pg-speaker-softmax-temp', type=float, default=1)
parser.add_argument('--failed-penalty', type=float, default=0.5)

args = parser.parse_args()