import numpy as np
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False

class Sampler:
    """Base importance sampler

    Parameters
    ----------
    n: int, number of data points.
    rs: int or RandomState, determines the random number generation.
    p: ndarray, a probability distribution over the points.
    """

    def __init__(self, n, rs, p=None):
        self.n = n
        self.rs = rs
        if p is None:
            self.p = np.ones(self.n)
            self.p /= n
        else:
            self.p = p

    def sample(self, batch_size):
        """Samples a batch of points

        Parameters
        ----------
        batch_size: int, the size of the batch.

        Returns
        -------
        A tuple of the selected points' indices and importance weights.
        """
        choice = self.rs.choice(self.n, batch_size, p=self.p)
        return choice, 1. / self.p[choice] / self.n

    def update(self, loss):
        pass


class VRM:
    """Variance reducer importance sampler based on a mixture of sampling distributions.

    Parameters
    ----------
    n: int, the number of data points.
    samplers: list of Samplers, the base samplers forming the mixture components.
    rs: int or RandomState, determines the random number generation.
    method: {'QP' or 'PGD'} the method for solving the projection step to the restricted simplex.
        'QP': the projection is solved by quadratic programming using cvxopt
        'PGD': the projection is solved by projected gradient descent.
    beta, gamma, eps: floats, the online Newton step's parameters. See paper for details.
    scale: float, the scaling of the loss function.
    nr_gd_step: int, number of gradient steps for the PGD solver.
    gd_step_size: float, the step size for the PGD solver.
    """

    def __init__(self, n, samplers, rs, method='QP', beta=10, gamma=0.01, eps=0.1, scale=1e14, nr_gd_step=5,
                 gd_step_size=0.2):
        self.n = n
        self.samplers = samplers
        self.rs = rs
        self.beta = beta
        self.gamma = gamma
        self.nr_samplers = len(samplers) + 1
        self.method = method

        # add uniform sampler
        self.samplers.append(Sampler(n, rs))
        self.w = np.ones(self.nr_samplers) * 1. / self.nr_samplers
        self.g = np.zeros(self.nr_samplers)

        self.scale = scale
        self.A = np.eye(self.nr_samplers) * eps
        self.A_inv = np.eye(self.nr_samplers) / eps
        self.c_G = matrix(-np.eye(self.nr_samplers))
        self.c_h = np.zeros(self.nr_samplers)
        self.c_h[-1] = -gamma
        self.c_h = matrix(self.c_h)
        self.c_A = matrix(np.ones(self.nr_samplers), (1, self.nr_samplers))
        self.c_b = matrix(1.0)

        self.nr_gd_step = nr_gd_step
        self.gd_step_size = gd_step_size

    def sample(self, batch_size):
        """Samples a batch of points

        Parameters
        ----------
        batch_size: int, the size of the batch.

        Returns
        -------
        A tuple of the selected points' indices and importance weights.
        """
        choice = self.rs.choice(np.arange(self.nr_samplers), p=self.w)
        X, _ = self.samplers[choice].sample(batch_size)

        # get the probability of the samples
        self.p_vec = np.zeros((self.nr_samplers, batch_size))
        for i in range(self.nr_samplers):
            self.p_vec[i] = self.samplers[i].p[X]
        self.p = self.w.dot(self.p_vec)
        return X, 1. / self.p / self.n

    def update(self, loss_sq):
        """Updates the sampler using the online Newton step.

        Parameters
        ----------
        loss_sq: float, the squared loss.
        """
        grad = - np.sum(self.p_vec * loss_sq / self.p ** 3 / self.scale, axis=1)
        self.A = self.A + np.outer(grad, grad)
        aux = np.dot(self.A_inv, grad)
        self.A_inv = self.A_inv - np.outer(aux, aux) / (1. + np.dot(grad, aux))
        aux = self.w - np.dot(self.A_inv, grad) / self.beta

        if self.method == 'QP':
            c_Q = matrix(self.A)
            c_p = matrix(- np.dot(self.A, aux))
            self.w = np.array(solvers.qp(c_Q, c_p, self.c_G, self.c_h, self.c_A, self.c_b)['x']).reshape(-1)
            self.w[self.w < 0] = 0
        elif self.method == 'PGD':
            for it in range(self.nr_gd_step):
                self.w = self.w - self.gd_step_size * np.dot(self.A, self.w - aux)
                self.w = self.project(self.w, 1)
                if self.w[-1] < self.gamma:
                    self.w[-1] = self.gamma
                    self.w[:-1] = self.project(self.w[:-1], 1 - self.gamma)
        else:
            raise Exception('Unknown projection method.')

        self.w /= np.sum(self.w)

    def project(self, w, z):
        """Projects onto the simplex.

        Parameters
        ----------
        w: ndarray, vector to project.
        z: the L1-norm of the simplex.

        Returns
        -------
            ndarray, the projected vector.
        """
        k = w.shape[0]
        u = np.sort(w)[::-1]
        cumsum = np.cumsum(u)
        v = (z - cumsum) / (np.arange(k) + 1) + u
        rho = np.searchsorted(v[::-1], 0, side='right')
        rho = k - rho - 1
        alpha = (z - cumsum[rho]) / (rho + 1)
        w = np.maximum(w + alpha, 0)
        return w
