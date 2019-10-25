from __future__ import print_function
import torch.utils.data as data
import numpy as np
import settings
from utils import generic_utils
import torch
import math
import matplotlib.pyplot as plt
from random import randint


class SYNTHETIC(data.Dataset):
    def __init__(self, partition='train', max_len=1000, tr_tt=1000, val_tt=1000, test_tt=1000, equations="acceleration", gnn_format=False, sparse=True, x0_format='v'):
        self.partition = partition  # training set or test set
        self.max_len = max_len
        self.equations = equations
        assert (equations == "acceleration" or equations == "velocity" or equations == "air_resistance")
        self.gnn_format = gnn_format
        self.sparse = sparse
        self.equations = equations
        if equations == "acceleration":
            self.x0 = settings.x0_a
            self.P0 = np.eye(len(self.x0))
        elif equations == "velocity":
            self.x0 = settings.x0_v
            self.P0 = np.eye(len(self.x0))
        elif equations == "air_resistance":
            self.x0 = settings.x0_a
            self.P0 = np.eye(len(self.x0))
        self.P0 = self.P0*1000

        seeds = {'test': 0, 'train': 50, 'val': 51}



        if self.partition == 'train':
            self.data = self._generate_sample(seeds[self.partition], tt=tr_tt, start_after=1000)
        elif self.partition == 'val':
            self.data = self._generate_sample(seeds[self.partition], tt=val_tt, start_after=1000)
        elif self.partition == 'test':
            self.data = self._generate_sample(seeds[self.partition], tt=test_tt, start_after=0)
        else:
            raise Exception('Wrong partition')
        self._split_data()

        self._generate_operators()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (state, meas) where target is index of the target class.
        """
        state, meas, x0, operator = self.data[index]
        pos = generic_utils.state2position(state).astype(np.float32)
        #x0, P0 = self.__pos2x0(pos[0, :])
        if self.gnn_format:
            return np.arange(0, meas.shape[0], 1), pos, meas.astype(np.float32), x0, self.P0, operator
        else:
            return state, meas,  x0, self.P0

    def __len__(self):
        return len(self.data)

    def __pos2x0(self, pos):
        if self.equations == 'acceleration':
            x0 = np.zeros(6).astype(np.float32)
            x0[0] = pos[0]
            x0[3] = pos[1]
            P0 = np.eye(6)*0
        elif self.x0_format == 'velocity':
            x0 = np.zeros(4).astype(np.float32)
            x0[0] = pos[0]
            x0[2] = pos[1]
            P0 = np.eye(4)*0
        else:
            raise Exception('Unvalid x0_format')
        return x0, P0

    def _split_data(self):
        num_splits = math.ceil(float(self.data[0].shape[0])/self.max_len)
        data = []
        for i in range(int(num_splits)):
            i_start = i*self.max_len
            i_end = (i+1)*self.max_len
            data.append([self.data[0][i_start:i_end], self.data[1][i_start:i_end], self.data[0][i_start]])
        self.data = data

    def _generate_operators(self):
        for i in range(len(self.data)):
            tt = self.data[i][0].shape[0]
            self.data[i].append(self.__buildoperators_sparse(tt))

    def _generate_sample(self, seed, tt, start_after=1000):
        np.random.seed(seed)
        if self.equations == "acceleration":
            sample = simulate_system(create_model_parameters_a, K=tt, x0=self.x0, start_after=start_after)
        elif self.equations == "velocity":
            sample = simulate_system(create_model_parameters_v, K=tt, x0=self.x0, start_after=start_after)
        elif self.equations == "air_resistance":
            sample = simulate_system(create_model_parameters_ar, K=tt, x0=self.x0, start_after=start_after)
        return list(sample)

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
        for state, meas, _, _ in self.data:
            total += meas.shape[0]
        return total


def create_model_parameters_v(T=1., s2_x = 0.15 **2, s2_y = 0.15 **2, lambda2=0.5 **2):

    # Motion model parameters

    F = np.array([[1, T],
             [0, 1]])
    base_sigma = np.array([[T, 0],
                          [0, T]])
    zeros_2 = np.zeros((2, 2))
    H = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])

    sigma_x = s2_x * base_sigma
    sigma_y = s2_y * base_sigma

    A = np.block([[F, zeros_2],
                  [zeros_2, F]])
    Q = np.block([[sigma_x, zeros_2],
                  [zeros_2, sigma_y]])

    # Measurement model parameters

    R = lambda2 * np.eye(2)

    return A, H, Q, R


def create_model_parameters_a(T=1., s2_x=0.1 ** 2, s2_y=0.1 ** 2, lambda2=0.5 ** 2, c=0.06, tau = 0.17):
    # Motion model parameters

    F = np.array([[1, T - 0.5 * c * T **2, 0.5 * T ** 2],
                  [0, 1 - c*T + 0.5* (c**2-tau) * (T**2), T - 0.5 * c * (T ** 2)],
                  [0, -tau+0.5*c*tau*T**2, 1-0.5*tau*T**2]])
    #print(F)
    #print("")
    base_sigma = np.array([[T / 3, 0, 0],
                        [0, T, 0],
                           [0, 0, 3*T]])

    sigma_x = s2_x * base_sigma
    sigma_y = s2_y * base_sigma

    zeros_2 = np.zeros((3, 3))
    A = np.block([[F, zeros_2],
                  [zeros_2, F]])

    Q = np.block([[sigma_x, zeros_2],
                  [zeros_2, sigma_y]])

    # Measurement model parameters
    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0]])
    R = lambda2 * np.eye(2)

    return (A, H, Q, R)


def simulate_system(parameters_creator, K, x0, start_after=0):
    K = K + start_after

    (A, H, Q, R) = parameters_creator()

    motion_model = MotionModel(A, Q)
    meas_model = MeasurementModel(H, R)

    (m, _) = Q.shape
    (n, _) = R.shape

    state = np.zeros((K, m))
    meas = np.zeros((K, n))

    # initial state
    x = x0
    for k in range(K):
        x = motion_model(x)
        z = meas_model(x)

        state[k, :] = x
        meas[k, :] = z

    return state[start_after:], meas[start_after:]


class MotionModel():
    def __init__(self, A, Q):
        self.A = A
        self.Q = Q

        (m, _) = Q.shape
        self.zero_mean = np.zeros(m)

    def __call__(self, x):
        new_state = self.A @ x + np.random.multivariate_normal(self.zero_mean, self.Q)
        return new_state


class MeasurementModel():
    def __init__(self, H, R):
        self.H = H
        self.R = R

        (n, _) = R.shape
        self.zero_mean = np.zeros(n)

    def __call__(self, x):
        measurement = self.H @ x + np.random.multivariate_normal(self.zero_mean, self.R)
        return measurement

def plot_trajecotry(positions):
    plt.figure()
    pos = positions[0]
    pred = positions[1]
    plt.plot(pos[:, 0], pos[:, 1], 1, linewidth=1)
    plt.scatter(pred[:, 0], pred[:, 1], 1, linewidth=1, c='r')
    plt.axis('equal')
    plt.title('By altitude')
    plt.show()

if __name__ == '__main__':
    dataset_test = SYNTHETIC(partition='test', tr_tt=1000, val_tt=1000, test_tt=1000,
                                         equations="acceleration", gnn_format=True, max_len=10000)
    plot_trajecotry([generic_utils.state2position(dataset_test.data[0][0]), dataset_test.data[0][1]])

