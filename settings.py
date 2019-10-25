import argparse
import numpy as np

def get_settings():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--exp_name', type=str, default='exp1')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--K', type=int, default=100, metavar='N',
                        help='number of iterations per trajectory')
    parser.add_argument('--tr_samples', type=int, default=100)
    parser.add_argument('--val_samples', type=int, default=100)
    parser.add_argument('--test_samples', type=int, default=100)
    parser.add_argument('--test_every', type=int, default=1, help='test every x epochs')
    parser.add_argument('--nclt_ratio', type=float, default=1.0)
    parser.add_argument('--nf', type=int, default=48)
    parser.add_argument('--model', type=str, default='kalman_smoother', metavar='N',
                        help='kalman_filter' +
                             'kalman_smoother')
    parser.add_argument('--sparse', type=bool, default=True)
    parser.add_argument('--learned', type=bool, default=False)
    parser.add_argument('--prior', type=bool, default=True)
    parser.add_argument('--taylor_K', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.005)
    parser.add_argument('--init', type=str, default='meas_invariant', help='messages; meas; meas_invariant')

    args = parser.parse_args()
    return args

x0_v = np.array([0, 0.1, 0, 0.1], dtype=np.float32)
x0_a = np.array([0, 0.1, 0, 0, 0.1, 0], dtype=np.float32)
