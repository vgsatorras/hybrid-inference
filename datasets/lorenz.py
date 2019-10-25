import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import torch.utils.data as data
import torch
import os
import pickle
from mpl_toolkits.mplot3d import Axes3D
import math

compact_path_005 = "temp/lorenz.pickle"
compact_path_001 = "temp/lorenz_001.pickle"
#compact_path = "../temp/lorenz.pickle"


class LORENZ(data.Dataset):
    def __init__(self, partition='train', max_len=1000, tr_tt=1000, val_tt=1000, test_tt=1000, gnn_format=False, sparse=True, sample_dt=0.05, no_pickle=False):
        self.partition = partition  # training set or test set
        self.max_len = max_len
        self.gnn_format = gnn_format
        self.sparse = sparse
        self.lamb = 0.5
        self.x0 = [1.0, 1.0, 1.0]
        self.H = np.diag([1]*3)
        self.R = np.diag([1]*3) * self.lamb ** 2
        self.sample_dt = sample_dt
        self.dt = 0.00001

        self.rho = 28.0
        self.sigma = 10.0
        self.beta = 8.0 / 3.0

        if self.sample_dt == 0.01:
            compact_path = compact_path_001
        else:
            compact_path = compact_path_005

        if no_pickle:
            self.data = self._generate_sample(seed=0, tt=tr_tt + val_tt + test_tt)
            '''elif self.sample_dt == 0.01 and not os.path.exists(compact_path):
                samples, meas = self._generate_sample(seed=0, tt=109000)
                samples = samples[4000:]
                meas = meas[4000:]
                self.data = [samples, meas]
                self.dump(compact_path, self.data)'''
        elif not os.path.exists(compact_path):
            self.data = self._generate_sample(seed=0, tt=204000)
            self.dump(compact_path, self.data)
        else:
            self.data = self.load(compact_path)


        if self.partition == 'test':
            self.data = [self.data[0][0:test_tt], self.data[1][0:test_tt]]
        elif self.partition == 'val':
            self.data = [self.data[0][test_tt:(test_tt+val_tt)], self.data[1][test_tt:(test_tt+val_tt)]]
        elif self.partition == 'train':
            self.data = [self.data[0][(test_tt+val_tt):(test_tt + val_tt + tr_tt)], self.data[1][(test_tt+val_tt):(test_tt + val_tt + tr_tt)]]
        else:
            raise Exception('Wrong partition')
        self._split_data()
        self._generate_operators()

        '''
        tr_samples = int(tr_tt/max_len)
        test_samples = int(test_tt / max_len)
        val_samples = int(val_tt / max_len)
        if self.partition == 'train':
            self.data = [self._generate_sample(i, max_len) for i in range(test_samples, test_samples + tr_samples)]
        elif self.partition == 'val':
            self.data = [self._generate_sample(i, max_len) for i in range(test_samples + tr_samples, test_samples + tr_samples + val_samples)]
        elif self.partition == 'test':
            self.data = [self._generate_sample(i, max_len) for i in range(test_samples)]
        else:
            raise Exception('Wrong partition')
        '''

        print("%s partition created, \t num_samples %d \t num_timesteps: %d" % (
        self.partition, len(self.data), self.total_len()))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (state, meas) where target is index of the target class.
        """
        state, meas, operator = self.data[index]
        x0 = state[0]
        P0 = np.eye(x0.shape[0])
        if self.gnn_format:
            return np.arange(0, meas.shape[0], 1), state.astype(np.float32), meas.astype(np.float32), x0, P0, operator
        else:
            return state, meas,  self.x0, self.P0

    def __len__(self):
        return len(self.data)

    def dump(self, path, object):
        if not os.path.exists('temp'):
            os.makedirs('temp')
        with open(path, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(path, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            return pickle.load(f)

    def _split_data(self):
        num_splits = math.ceil(float(self.data[0].shape[0])/self.max_len)
        data = []
        for i in range(int(num_splits)):
            i_start = i*self.max_len
            i_end = (i+1)*self.max_len
            data.append([self.data[0][i_start:i_end], self.data[1][i_start:i_end]])
        self.data = data

    def _generate_operators(self):
        for i in range(len(self.data)):
            tt = self.data[i][0].shape[0]
            self.data[i].append(self.__buildoperators_sparse(tt))

    def _generate_sample(self, seed, tt):
        np.random.seed(seed)
        sample = self._simulate_system(tt=tt, x0=self.x0)

        # returns state, measurement
        return list(sample)

    def f(self, state, t):
        x, y, z = state  # unpack the state vector
        return self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z  # derivatives

    def _simulate_system(self, tt, x0):
        t = np.arange(0.0, tt*self.sample_dt, self.dt)
        states = odeint(self.f, x0, t)
        states_ds = np.zeros((tt, 3))
        for i in range(states_ds.shape[0]):
            states_ds[i] = states[i*int(self.sample_dt/self.dt)]
        states = states_ds

        #Measurement
        meas_model = MeasurementModel(self.H, self.R)
        meas = np.zeros(states.shape)
        for i in range(len(states)):
            meas[i] = meas_model(states[i])
        return states, meas


    def __build_operators(self, nn=20):
        # Identity
        I = np.expand_dims(np.eye(nn), 2)

        #Messages
        mr = np.pad(I, ((1, 0), (0, 0), (0, 0)), 'constant', constant_values=(0))[0:nn, :, :]
        ml = np.pad(I, ((0, 1), (0, 0), (0, 0)), 'constant', constant_values=(0))[1:(nn+1), :, :]

        return np.concatenate([I, mr, ml], axis=2).astype(np.float32)

    def __buildoperators_sparse_old(self, nn=20):
        # Identity
        i = torch.LongTensor([[i,i] for i in range(nn)])
        v = torch.FloatTensor([1 for i in range(nn)])
        I = torch.sparse.FloatTensor(i.t(), v)

        #Message right
        i = torch.LongTensor([[i, i+1] for i in range(nn-1)] + [[nn-1, nn-1]])
        v = torch.FloatTensor([1 for i in range(nn-1)] + [0])
        mr = torch.sparse.FloatTensor(i.t(), v)

        #Message left
        i = torch.LongTensor([[0, nn-1]] + [[i+1, i] for i in range(nn-1)])
        v = torch.FloatTensor([0] + [1 for i in range(nn-1)])
        ml = torch.sparse.FloatTensor(i.t(), v)

        return [I, mr, ml]

    def __buildoperators_sparse(self, nn=20):
        # Message right to left
        m_left_r = []
        m_left_c = []

        m_right_r = []
        m_right_c = []

        m_up_r = []
        m_up_c = []

        for i in range(nn - 1):
            m_left_r.append(i)
            m_left_c.append((i + 1))

            m_right_r.append(i + 1)
            m_right_c.append((i))

        for i in range(nn):
            m_up_r.append(i)
            m_up_c.append(i + nn)

        m_left = [torch.LongTensor(m_left_r), torch.LongTensor(m_left_c)]
        m_right = [torch.LongTensor(m_right_r), torch.LongTensor(m_right_c)]
        m_up = [torch.LongTensor(m_up_r), torch.LongTensor(m_up_c)]

        return {"m_left": m_left, "m_right": m_right, "m_up": m_up}


    def total_len(self):
        total = 0
        for state, meas, _ in self.data:
            total += meas.shape[0]
        return total


def __plot_trajectory(states):
    fig = plt.figure(linewidth=0.0)
    ax = fig.gca(projection='3d')
    ax.plot(states[:, 0], states[:, 1], states[:, 2], linewidth=0.5)
    plt.axis('off')
    plt.show()

def plot_trajectory(args, states, mse=0.):
    if not os.path.exists('plots'):
        os.makedirs('plots')

    if args.learned and args.prior:
        path = 'hybrid'
    elif args.learned and not args.prior:
        path = 'learned'
    elif not args.learned and args.prior:
        path = 'prior'
    else:
        path = 'baseline'

    if not os.path.exists('plots/%s' % path):
        os.makedirs('plots/%s' % path)

    fig = plt.figure(linewidth=0.0)
    ax = fig.gca(projection='3d')
    ax.plot(states[:, 0], states[:, 1], states[:, 2], linewidth=0.5)
    plt.axis('off')
    plt.savefig('plots/%s/tr_samples_%d_loss_%.4f.png' % (path, args.tr_samples, mse))

class MeasurementModel():
    def __init__(self, H, R):
        self.H = H
        self.R = R

        (n, _) = R.shape
        self.zero_mean = np.zeros(n)

    def __call__(self, x):
        measurement = self.H @ x + np.random.multivariate_normal(self.zero_mean, self.R)
        return measurement

if __name__ == '__main__':
    dataset = LORENZ(partition='test', sample_dt=0.01, no_pickle=False, max_len=5000, test_tt=5000, val_tt=0, tr_tt=0)
    __plot_trajectory(dataset.data[0][0])














