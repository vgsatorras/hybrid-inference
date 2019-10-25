import numpy as np
from filterpy.kalman import KalmanFilter as KF
from filterpy.kalman import ExtendedKalmanFilter
import math

class KalmanFilter():
    def __init__(self, A, H, Q, R, x_0, P_0):
        #Init Kalman filter
        self.fk = KF(dim_x=len(x_0), dim_z=H.shape[0])

        #Initial state
        self.fk.x = x_0  # state (x and dx)
        self.fk.P = P_0  # covariance matrix

        self.x_0 = None
        self.P_0 = None
        self.x_len = A.shape[0]

        #Model parameters
        self.fk.F = A # state transition matrix
        self.fk.H = H  # Measurement function
        self.fk.R = R  # state uncertainty
        self.fk.Q = Q  # process uncertainty

    def forward(self, meas, x_0, P_0):
        x_0, P_0 = self.adapt_x0(x_0, P_0)
        self.fk.x = x_0
        self.fk.P = P_0
        mu, cov, _, _ = self.fk.batch_filter(meas)
        return mu, cov

    def adapt_x0(self, x_0, P_0):
        if self.fk.F.shape[0] < x_0.shape[0]:
            x_0 = np.concatenate([x_0[0:2], x_0[3:5]])
            P_0 = P_0[0:4, 0:4]
        return x_0, P_0



class KalmanSmoother():
    def __init__(self, A, H, Q, R, x_0, P_0):
        self.kf = KalmanFilter(A, H, Q, R, x_0, P_0)

    def forward(self, meas, x_0, P_0):
        mu, cov = self.kf.forward(meas, x_0, P_0)
        M, P, C, _ = self.kf.fk.rts_smoother(mu, cov)
        return M, P


class ExtendedKalman_lorenz():
    def __init__(self, K, dim_x=3, dim_z=3, lamb=2, sigma=2, dt=0.05):
        #Init Kalman filter
        self.rk = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.rk.x = np.array([1, 1, 1])
        self.sigma = sigma
        self.lamb = lamb
        self.K = K

        self.rho = 28.0
        self.sigma_lorenz = 10.0
        self.beta = 8.0 / 3.0

        self.dt = dt
        self.rk.F = self.get_F(self.rk.x)
        self.rk.R = np.diag([lamb**2]*3)
        self.rk.Q = self.get_Q(dt=self.dt)
        self.rk.H = np.identity(3, dtype=np.float32)

        self.rk.P *= 3


    def get_F(self, x):
        A_ = np.array([[-self.sigma_lorenz, self.sigma_lorenz, 0],
                               [self.rho - x[2], -1, 0],
                               [x[1], 0, -self.beta]], dtype=np.float32)

        A = np.diag([1]*3).astype(np.float32)
        for i in range(1, self.K+1):
            if i == 1:
                A_p = A_
            else:
                A_p = np.matmul(A_, A_p)
            new_coef = A_p * np.power(self.dt, i) / float(math.factorial(i))
            A += new_coef

        return A

    def get_Q(self, dt):
        Q = np.diag([dt]*3)*(self.sigma**2)
        return Q

    def set_tran_eq(self, dt):
        self.rk.F = self.get_F(dt=dt)
        self.rk.Q = self.get_Q(dt=dt)

        return self.rk.F, self.rk.Q

    def HJacobian_at(self, x):
        """ compute Jacobian of H matrix at x """

        H_J = np.identity(3, dtype=np.float32)

        return H_J

    def hx(self, x):
        return x

    def rts_smoother(self, Xs, Ps, Fs=None, Qs=None, inv=np.linalg.inv):
        if len(Xs) != len(Ps):
            raise ValueError('length of Xs and Ps must be the same')

        n = Xs.shape[0]
        dim_x = Xs.shape[1]

        if Fs is None:
            Fs = [self.A] * n
        if Qs is None:
            Qs = [self.rk.Q] * n

        # smoother gain
        K = np.zeros((n, dim_x, dim_x))

        x, P, pP = Xs.copy(), Ps.copy(), Ps.copy()
        x_new, P_new, pP_new = Xs.copy(), Ps.copy(), Ps.copy()
        for k in range(n - 2, -1, -1):
            pP_new[k] = dot(dot(Fs[k], P[k]), Fs[k].T) + Qs[k]

            # pylint: disable=bad-whitespace
            K[k] = dot(dot(P[k], Fs[k].T), np.linalg.inv(pP_new[k]))
            x_new[k] += dot(K[k], x[k + 1] - dot(Fs[k], x[k]))
            P_new[k] += dot(dot(K[k], P[k + 1] - pP_new[k]), K[k].T)

        return (x_new, P_new, K, pP_new)

    def forward(self, observations):
        self.rk.x = observations[0]
        xs, track, Ps, Fs = [], [], [], []
        for obs in observations:
            track.append(obs)

            self.rk.update(obs, self.HJacobian_at, self.hx)
            xs.append(self.rk.x)
            Ps.append(self.rk.P)
            self.rk.F = self.get_F(self.rk.x)
            Fs.append(self.rk.F)
            self.rk.predict()

        xs = np.array(xs)

        M, P, C, _ = self.rts_smoother(xs, Ps, Fs)

        return M




def dot(a, b):
    return a @ b
