import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn.modules.module import Module
import math


class MLP(nn.Module):
    """Three-layer fully-connected RELU net"""

    def __init__(self, n_in, n_hid, n_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.bn = nn.BatchNorm1d(n_out)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]

        x = self.fc1(inputs)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        return x

class Graph_bn(nn.Module):
    def __init__(self, nf):
        super(Graph_bn, self).__init__()
        self.nf = nf
        self.bn = nn.BatchNorm1d(nf)

    def forward(self, x):
        x = x.transpose(1, 2)
        x_size = x.size()
        x = x.contiguous()
        x = x.view(-1, self.nf)
        x = self.bn(x)
        x = x.view(*x_size[:-1], self.nf)
        x = x.transpose(1, 2)
        return x


class G_E_GNN(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, nf, dim_state, dim_meas, hybrid=True):
        super(G_E_GNN, self).__init__()
        self.hybrid = hybrid
        self.nf = nf
        self.dim_meas = dim_meas
        act_fn = nn.ReLU()

        if not self.hybrid:
            dim_state_edge = 0
            dim_meas_edge = 0
        else:
            dim_state_edge = dim_state
            dim_meas_edge = dim_meas

        self.fc_init_y = nn.Conv1d(self.dim_meas*2, self.nf, kernel_size=1)

        self.edge_mlp_l = MLP(nf*2 + dim_state_edge, nf, nf)
        self.edge_mlp_r = MLP(nf * 2 + dim_state_edge, nf, nf)
        self.edge_mlp_u = MLP(nf * 2 + dim_state_edge, nf, nf)


        self.node_mlp = nn.Sequential(
            nn.Linear(nf, nf),
            nn.BatchNorm1d(nf),
            nn.LeakyReLU(),
            )

        self.gru = nn.GRUCell(nf, nf)

        self.h2state_1 = nn.Conv1d(self.nf, self.nf, kernel_size=1)
        self.dec_bn = Graph_bn(self.nf)

        if self.hybrid:
            self.h2state_2 = nn.Conv1d(self.nf, int(dim_state), kernel_size=1)
            torch.nn.init.xavier_uniform_(self.h2state_2.weight, gain=0.1)
            torch.nn.init.uniform_(self.h2state_2.bias, a=-0.02, b=0.02)

        else:
            self.h2state_2 = nn.Conv1d(self.nf, int(dim_meas), kernel_size=1)

    def edge_model(self, source, target, edge_attr, row, batch, edge_type):
        del batch  # Unused.
        if self.hybrid:
            out = torch.cat([source, target, edge_attr], dim=1)
        else:
            out = torch.cat([source, target], dim=1)

        if edge_type == "m_left":
            out = self.edge_mlp_l(out)
        elif edge_type == "m_right":
            out = self.edge_mlp_r(out)
        elif edge_type == "m_up":
            out = self.edge_mlp_u(out)
        else:
            raise("Error")
        return out

    def node_model(self, h, row, edge_feat, node_attr, u, batch):
        # Sums over column indexes
        del u, batch  # Unused.
        agg = unsorted_segment_sum(edge_feat, row, num_segments=h.size(0))
        agg = self.node_mlp(agg)
        h = self.gru(agg, h)
        return h

    def init_h(self, meas):
        meas_inv = meas2inv(meas)

        self.hy = self.fc_init_y(meas_inv).squeeze(0).transpose(0,1)  # self.layer0(meas_inv)

        h = torch.randn((meas_inv.size(0), self.nf, meas_inv.size(2)))
        return h

    def decode(self, h):
        h = self.h2state_1(h)
        return self.h2state_2(F.relu(self.dec_bn(h)))

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, u=None, batch=None):
        #Reshaping nodes
        h = h.squeeze(0).transpose(0, 1)
        length = h.shape[0]
        h = torch.cat([h, self.hy])

        #Reshaping edges
        edge_attr = [edge.squeeze(0).transpose(0, 1) for edge in edge_attr]

        if len(edge_attr) > 0:
            edge_attr = {"m_right": edge_attr[0][1:], "m_up": edge_attr[1], "m_left": edge_attr[2][:-1]}
        else:
            edge_attr = {"m_right": None, "m_up": None, "m_left": None}

        edge_feat = []
        rows = []
        for key in edge_index:

            row, col = edge_index[key]
            edge_feat.append(self.edge_model(h[row], h[col], edge_attr[key], row, batch, edge_type=key))
            rows.append(row)
        h = h[:length]
        h = self.node_model(h, torch.cat(rows), torch.cat(edge_feat), node_attr, u, batch)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN

        h = h.transpose(0, 1).unsqueeze(0)
        return self.decode(h), h


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


class GNN_Kalman(nn.Module):
    def __init__(self, args, A, H, Q, R, x_0, P_0, nf,  prior=True, learned=True, init='meas_invariant', gamma=0.005):
        super(GNN_Kalman, self).__init__()
        '''
        meas_format: messages, meas, meas_invariant
        '''
        # Model parameters
        self.args = args
        self.A = self.np2torch(A)
        self.H = self.np2torch(H)
        self.Q = self.np2torch(Q)
        self.Q_inv = self.np2torch(np.linalg.inv(Q))
        self.R = self.np2torch(R)
        self.R_inv = self.np2torch(np.linalg.inv(R))
        self.init = init

        # Initial state
        self.x_0_np = x_0
        self.x_0 = self.np2torch(x_0)
        self.P_0 = self.np2torch(P_0)

        self.gamma = gamma

        self.trainable = {'A': False, 'H': False, 'Q': False, 'R': False}

        #GNN parameters
        self.nf = nf
        self.J = 3
        self.dim_meas = self.H.shape[1]
        self.dim_state = self.A.shape[1]
        #self.dim_input = self.dim_state + self.dim_meas
        self.alpha = torch.nn.Parameter(torch.ones(1))

        if self.init == 'messages':
            self.dim_input = self.dim_state
            self.extra_input_gate = 0
        elif self.init == 'meas':
            self.dim_input = self.dim_meas
            self.extra_input_gate = self.dim_meas + self.dim_state
        elif self.init == 'meas_invariant':
            self.dim_meas = self.dim_meas
            self.dim_input = self.dim_meas
            self.extra_input_gate = self.dim_meas
        else:
            raise Exception('incorrect meas_format')
        self.prior = prior
        self.learned = learned

        self.gnn = G_E_GNN(self.nf, self.dim_state, self.dim_meas, hybrid=prior)

    def _gen_params(self, variable):
        size = list(variable.size())
        size[0] = 1  # force bs = 1
        return nn.Parameter(torch.randn(size, device=self.args.device, requires_grad=True))

    def set_trainable_bools(self, matrices=[]):
        for name in matrices:
            self.trainable[name] = True
        self._gen_params_all()

    def np2torch(self, a):
        a = torch.from_numpy(a.astype(np.float32))
        a = torch.unsqueeze(a, 0)
        return a.to(self.args.device)

    def _gen_params_all(self):
        for name in self.trainable:
            if self.trainable[name]:
                if name == 'A':
                    self.A_params = self._gen_params(self.A)
                elif name == 'H':
                    self.H_params = self._gen_params(self.H)
                elif name == 'Q':
                    self.Q_inv_params = nn.Parameter(torch.randn(self.Q_inv.size(-1), device=self.args.device, requires_grad=True))
                elif name == 'R':
                    self.R_inv_params = self._gen_params(self.R_inv)

    def _set_trainable(self):
        for name in self.trainable:
            if self.trainable[name]:
                if name == 'A':
                    self.A = torch.abs(self.A_params)
                elif name == 'H':
                    self.H = torch.abs(self.H_params)
                elif name == 'Q':
                    Q_inv = torch.diag(torch.abs(self.Q_inv_params)).unsqueeze(0)
                    self.Q_inv = Q_inv
                elif name == 'R':
                    self.R_inv = torch.abs(self.R_inv_params)


    def update_bs(self, tensor, bs):
        repetitions = [1] * len(tensor.size())
        repetitions[0] = bs
        tensor = tensor.repeat(repetitions)
        return tensor

    def update_bs_all(self, bs):
        self.A_b = self.update_bs(self.A, bs)
        self.H_b = self.update_bs(self.H, bs)
        self.Q_b = self.update_bs(self.Q, bs)
        self.Q_inv_b = self.update_bs(self.Q_inv, bs)
        self.R_b = self.update_bs(self.R, bs)
        self.R_inv_b = self.update_bs(self.R_inv, bs)
        self.x_0_b = self.update_bs(self.x_0, bs)

    def m1(self, x):
        "Message from x_{t-1} to x_{t}"
        x_t_m1 = torch.cat([x[:, :, [0]], x], dim=2)[:, :, 0:x.size(2)].clone()
        #x_t_m1 = torch.cat([x0.unsqueeze(-1), x], dim=2)[:, :, 0:x.size(2)].clone()
        pred = torch.bmm(self.A_b, x_t_m1)
        m1 = - torch.bmm(self.Q_inv_b, x - pred - self.u_m1())
        m1 = m1[:, :, 1:(x.size(-1))].clone()
        m1 = F.pad(m1, (1, 0), mode='constant', value=0)
        return m1

    def m2(self, x, meas):
        "Message from y_t to x_t"
        coef1 = torch.bmm(self.H_b.transpose(1, 2), self.R_inv_b)
        coef2 = (meas - torch.bmm(self.H_b, x))
        m2 = torch.bmm(coef1, coef2)
        # (self.H.transpose() @ self.R_inv) @ (meas - self.H @ x)
        return m2

    def m3(self, x):
        "Message from x_{t+1} to x_t"
        x_t_p1 = F.pad(x, (0, 1), mode='constant', value=0)
        x_t_p1 = x_t_p1[:, :, 1:(x.size(2)+1)].clone()
        coef1 = torch.bmm(self.A_b.transpose(1, 2), self.Q_inv_b)
        coef2 = x_t_p1 - torch.bmm(self.A_b, x) - self.u_p1()
        m3 = torch.bmm(coef1, coef2)
        m3 = m3[:, :, 0:(x.size(-1) - 1)].clone()
        m3 = F.pad(m3, (0, 1), mode='constant', value=0)
        return m3

    def u_m1(self):
        return self.u()

    def u_p1(self):
        return self.u()

    def u(self):
        return 0

    def p_messages(self, x, meas, x0):
        if not self.prior:
            return []
        else:
            m1 = self.m1(x).detach()
            m2 = self.m2(x, meas).detach()
            m3 = self.m3(x).detach()
            return [m1, m2, m3]

    def init_states(self, meas):
        in_state = torch.bmm(self.H_b.transpose(1, 2), meas)
        return torch.tensor(in_state.data)

    def state2pos(self, state):
        return torch.bmm(self.H_b, state)

    def forward(self, input, x0, T=50, ts=None):
        self.ts = ts
        [operators, meas] = input

        meas = torch.transpose(meas, 1, 2)
        self.update_bs_all(meas.size(0))
        x = self.init_states(meas)

        ## Init h from observations ##
        h = self.gnn.init_h(meas)

        pos_track = []
        for i in range(T):
            if self.prior:
                Mp_arr = self.p_messages(x, meas, x0)
            else:
                Mp_arr = []
            grad, h = self.gnn(h, operators, Mp_arr)

            if self.learned and self.prior:
                x = x + self.gamma * (grad + sum(Mp_arr))
                pos_track.append(self.state2pos(x).transpose(1, 2))
            elif self.learned and not self.prior:
                pred = grad + meas
                pos_track.append(pred.transpose(1, 2))
            elif not self.learned and self.prior:
                x = x + self.gamma * (grad*0 + sum(Mp_arr))
                pos_track.append(self.state2pos(x).transpose(1, 2))
            else:  # not self.learned and not self.prior
                pred = grad*0 + meas
                pos_track.append(pred.transpose(1, 2))

        return pos_track


class Hybrid_lorenz(GNN_Kalman):

    def __init__(self, args, sigma, lamb, nf, dt, K=1, prior=True, learned=True, init='meas_invariant', gamma=0.005):
        self.sigma = sigma
        self.lamb = lamb
        self.dt = dt
        self.K = K
        self.rho = 28.0
        self.sigma_lorenz = 10.0
        self.beta = 8.0 / 3.0
        x_0 = np.array([1., 1., 1.]).astype(np.float32)
        P_0 = np.diag([1] * 3) * 5
        A, Q, _ = self.gen_tran_matrices(x_0)
        H, R, _ = self.gen_meas_matrices()


        GNN_Kalman.__init__(self, args, A, H, Q, R, x_0, P_0, nf, prior=prior, learned=learned, init=init, gamma=gamma)

    def gen_tran_matrices(self, x):
        A_ = np.array([[-self.sigma_lorenz, self.sigma_lorenz, 0],
                               [self.rho - x[2], -1, 0],
                               [x[1], 0, -self.beta]], dtype=np.float32)

        sigma2 = self.sigma ** 2
        Q = sigma2 * np.diag([1]*3) * self.dt
        Q_inv = np.diag([1]*3) / (sigma2 * self.dt)

        A = np.diag([1]*3).astype(np.float32)
        for i in range(1, self.K+1):
            if i == 1:
                A_p = A_
            else:
                A_p = np.matmul(A_, A_p)
            new_coef = A_p * np.power(self.dt, i) / float(math.factorial(i))
            A += new_coef

        return A, Q, Q_inv

    def gen_meas_matrices(self):
        R = np.diag([self.lamb ** 2] * 3)
        R_inv = np.diag([1/(self.lamb ** 2)] * 3)
        H = np.diag([1]*3)
        return H, R, R_inv

    def update_bs(self, tensor, bs):
        repetitions = [1] * len(tensor.size())
        repetitions[0] = bs
        tensor = tensor.repeat(repetitions)
        return tensor

    def update_bs_A_Q(self, x):
        As, Qs, Q_invs = [], [], []
        for i in list(range(x.shape[2] - 1)):
            A, Q, Q_inv = self.gen_tran_matrices(x.detach().cpu().numpy()[0, :, i])
            As.append(self.np2torch(A))
            Qs.append(self.np2torch(Q))
            Q_invs.append(self.np2torch(Q_inv))
        return torch.cat(As, dim=0).to(self.args.device), torch.cat(Qs, dim=0).to(self.args.device), torch.cat(Q_invs, dim=0).to(self.args.device)

    def update_trans_model(self, x):
        self.A_b, self.Q_b, self.Q_inv_b = self.update_bs_A_Q(x)

    def update_meas_model(self, bs):
        self.H_b = self.update_bs(self.H, bs)
        self.R_b = self.update_bs(self.R, bs)
        self.R_inv_b = self.update_bs(self.R_inv, bs)
        self.x_0_b = self.update_bs(self.x_0, bs)

    def m1(self, x):
        A_b = torch.cat([self.A_b[[0]], self.A_b], dim=0)
        Q_inv_b = torch.cat([self.Q_inv_b[[0]], self.Q_inv_b], dim=0)
        x_t_m1 = torch.cat([x[:, :, [0]], x], dim=2)[:, :, 0:x.size(2)].clone()

        # x_t_m1.shape = (bs, nf, Nodes)
        x_t_m1 = x_t_m1.transpose(0, 2) # x_t_m1.shape = (nodes, nf, bs = 1)
        x = x.transpose(0, 2)
        pred = torch.bmm(A_b, x_t_m1)
        m1 = - torch.bmm(Q_inv_b, x - pred).transpose(0, 2)

        x = x.transpose(0, 2)
        m1 = m1[:, :, 1:(x.size(-1))].clone()
        m1 = F.pad(m1, (1, 0), mode='constant', value=0)
        return m1

    def m3(self, x):
        A_b = torch.cat([self.A_b, self.A_b[-1:]], dim=0)
        Q_inv_b = torch.cat([self.Q_inv_b, self.Q_inv_b[-1:]], dim=0)
        x_t_p1 = F.pad(x, (0, 1), mode='constant', value=0)
        x_t_p1 = x_t_p1[:, :, 1:(x.size(2)+1)].clone()
        coef1 = torch.bmm(A_b.transpose(1, 2), Q_inv_b)
        x_t_p1 = x_t_p1.transpose(0, 2)
        x = x.transpose(0, 2)
        coef2 = x_t_p1 - torch.bmm(A_b, x) - self.u_p1()
        m3 = torch.bmm(coef1, coef2)
        m3 = m3.transpose(0, 2)
        x = x.transpose(0, 2)
        m3 = m3[:, :, 0:(x.size(-1) - 1)].clone()
        m3 = F.pad(m3, (0, 1), mode='constant', value=0)
        return m3

    def forward(self, input, x0, T=20, ts=None):
        self.ts = ts
        [operators, meas] = input
        self._set_trainable()
        meas = torch.transpose(meas, 1, 2)
        self.update_meas_model(meas.size(0))
        x = self.init_states(meas)

        ## Init h from observations ##
        if self.init == 'messages':
            Mnp = self.p_messages(x, meas, x0)
            h = self.layer0(sum(Mnp))
        elif self.init == 'meas':
            h = self.layer0(meas)
        elif self.init == 'meas_invariant':
            h = self.gnn.init_h(meas)
        else:
            raise Exception('Error')


        pos_track = []
        for i in range(T):
            self.update_trans_model(x)
            if self.prior:
                Mp_arr = self.p_messages(x, meas, x0)
            else:
                Mp_arr = []
            grad, h = self.gnn(h, operators, Mp_arr)

            if self.learned and self.prior:
                x = x + self.gamma * (grad + sum(Mp_arr))
                pos_track.append(self.state2pos(x).transpose(1, 2))
            elif self.learned and not self.prior:
                pred = grad + meas
                pos_track.append(pred.transpose(1, 2))
            elif not self.learned and self.prior:
                x = x + self.gamma * (grad*0 + sum(Mp_arr))
                pos_track.append(self.state2pos(x).transpose(1, 2))
            else:  # not self.learned and not self.prior
                pred = grad*0 + meas
                pos_track.append(pred.transpose(1, 2))

        return pos_track


def meas2inv(meas):
    meas_l = F.pad(meas, (1, 0), mode='replicate', value=0)[:,:, :-1]
    meas_r = F.pad(meas, (0, 1), mode='replicate', value=0)[:,:, 1:]
    meas_l = meas - meas_l
    meas_r = meas_r - meas
    meas = torch.cat([meas_l, meas_r], 1)

    return meas