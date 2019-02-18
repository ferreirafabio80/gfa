import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.special import multigammaln, digamma


class VCCA(object):

    def __init__(self, d, N, a, b, beta, K, nu):

        self.d = d  # number of dimensions of data sources
        m = np.min(d)
        self.m = m   # number of different models
        self.N = N   # number of data points

        # Hyperparameters
        self.a = a
        self.b = b
        self.beta = beta
        self.K = K
        self.nu = nu

        # Variational parameters
        self.means_z = np.random.randn(m, N)
        self.sigma_z = np.random.randn(m, m)
        self.means_mu = [[] for _ in range(d.shape[1])]
        self.sigma_mu = [[] for _ in range(d.shape[1])]
        self.means_w = [[] for _ in range(d.shape[1])]
        self.sigma_w = [[] for _ in range(d.shape[1])]
        self.a_new = [[] for _ in range(d.shape[1])]
        self.b_new = [[] for _ in range(d.shape[1])]
        self.K_tilde = [[] for _ in range(d.shape[1])]
        self.nu_tilde = [[] for _ in range(d.shape[1])]
        for i in range(0, d.shape[1]):
            self.means_mu[i] = np.random.randn(d[i], 1)
            self.sigma_mu[i] = np.random.randn(d[i], d[i])
            self.means_w[i] = np.random.randn(d[i], m)
            self.sigma_w[i] = np.random.randn(m, m)
            self.a_new[i] = self.a + d[i]/2.0
            self.b_new[i] = np.abs(np.random.randn(m, 1))
            self.K_tilde[i] = np.abs(np.random.randn(d[i], d[i]))
            self.nu_tilde[i] = self.nu[i] + N

    def update_z(self, X):
        self.E_phi = [[] for _ in range(self.d.shape[1])]
        for i in range(0, self.d.shape[1]):
            self.E_phi[i] = np.dot(
                self.nu_tilde[i], np.linalg.inv(self.K_tilde[i]))

        self.sigma_z = np.linalg.inv(np.identity(self.m) +
                np.trace(np.dot(self.E_phi[0], self.sigma_w[0])) +
            self.means_w[0].T.dot(self.E_phi[0]).dot(self.means_w[0]) +
                np.trace(np.dot(self.E_phi[1], self.sigma_w[1])) +
            self.means_w[1].T.dot(self.E_phi[1]).dot(self.means_w[1]))
        self.means_z = self.sigma_z.dot(((X[0] - self.means_mu[0]).T.dot(
            self.E_phi[0]).dot(self.means_w[0]) + (X[1] -
            self.means_mu[1]).T.dot(self.E_phi[1]).dot(self.means_w[1])).T)

    def update_mu(self, X):
        for i in range(0, self.d.shape[1]):
            self.sigma_mu[i] = np.linalg.inv(self.beta[i] * np.identity(self.d[i]) +
                self.N * self.E_phi[i])
            S = 0
            for n in range(0, self.N):
                x_n = np.reshape(X[i][n, :].T, (self.d[i], 1))
                z_n = np.reshape(self.means_z[:, n].T, (self.m, 1))
                S += x_n - np.dot(self.means_w[i], z_n)
            self.means_mu[i] = self.E_phi[i].dot(self.sigma_mu[i], S)

    def update_w(self, X):
        for i in range(0, self.d.shape[1]):
            E_zz = self.sigma_z[i] + np.dot(self.means_z[i], self.means_z[i].T)
            for j in range(0, self.d[i]):
                self.sigma_w[i] = np.linalg.inv(np.diagonal(
                    self.a_new[i]/self.b_new[i]) + self.E_phi[i][j,j] * E_zz)
                for n in range(0, self.N):
                    S1 = 0
                    S2 = 0
                    if j <= self.m:
                        x_n_j = X[i][n, j]
                        z_n = np.reshape(self.means_z[:, n].T, (self.m, 1))
                        S1 += (x_n_j - self.means_mu[i][j]).T.dot(self.E_phi[i][:, j]).dot(z_n.T)
                    else:
                        S2 += E_zz * np.dot(self.means_w[i][j, :].T, self.E_phi[i][j, self.d[i]])
                S = (S1 - S2).T
                self.means_w[i][j] = np.reshape(
                    np.dot(self.sigma_w, S), self.m)

    def update_alpha(self):
        for i in range(0, self.d.shape[1]):
            for j in range(0, self.m):
                self.b_new[i][j] = self.b + (np.trace(self.sigma_w[i]) + np.dot(
                    self.means_w.T[i][:, j].T, self.means_w[i][:, j].T)) / 2

    def update_phi(self, X):
        for i in range(0, self.d.shape[1]):
            A = 0
            for n in range(0, self.N):
                A += np.dot(X[i][:, n].T, X[i][:, n])
                A -= np.dot(X[i][:, n].T, self.means_w[i]).dot(self.means_z[:, n])
                A -= np.dot(X[i][:, n].T, self.means_mu[i])
                A -= np.dot(self.means_w[i].T, self.means_z[:, n].T).dot(X[i][:, n])
                A += np.dot(self.means_w[i].T, self.means_z[:, n].T).dot(self.means_w[i]).dot(self.means_z[:, n])
                A += np.trace(self.sigma_w[i]) + np.dot(self.means_w[i].T, self.means_w[i]) * \
                     np.trace(self.sigma_z) + np.dot(self.means_z[:, n].T, self.means_z[:, n])
                A += np.dot(self.means_w[i].T, self.means_z[:, n].T).dot(self.means_mu[i])
                A -= np.dot(self.means_mu[i].T, X[i][:, n])
                A += np.dot(self.means_mu[i].T, self.means_w[i]).dot(self.means_z[:, n])
                A += np.trace(self.sigma_mu[i]) + np.dot(self.means_mu[i].T, self.means_mu[i])
            self.K_tilde[i] = self.K[i] + A

    def L(self, X):

        L = 0
        ###Terms from expectations###
        # N(X_n|Z_n)
        for i in range(0, self.d.shape[1]):
            S = 0
            L += -self.N/2 * (self.d[i] * np.log(2*np.pi) - multigammaln(self.nu_tilde[i]/2,self.d[i]) - \
                self.d[i] * np.log(2) - np.log(np.linalg.det(np.linalg.inv(self.K_tilde[i])))) + \
                    self.nu_tilde[i] * self.K_tilde[i]
            for n in range(0, self.N):
                S += np.dot(X[i][:, n].T, X[i][:, n])
                S -= np.dot(X[i][:, n].T, self.means_w[i]).dot(self.means_z[:, n])
                S -= np.dot(X[i][:, n].T, self.means_mu[i])
                S -= np.dot(self.means_w[i].T, self.means_z[:, n].T).dot(X[i][:, n])
                S += np.dot(self.means_w[i].T, self.means_z[:, n].T).dot(self.means_w[i]).dot(self.means_z[:, n])
                S += np.trace(self.sigma_w[i]) + np.dot(self.means_w[i].T, self.means_w[i]) * \
                     np.trace(self.sigma_z) + np.dot(self.means_z[:, n].T, self.means_z[:, n])
                S += np.dot(self.means_w[i].T, self.means_z[:, n].T).dot(self.means_mu[i])
                S -= np.dot(self.means_mu[i].T, X[i][:, n])
                S += np.dot(self.means_mu[i].T, self.means_w[i]).dot(self.means_z[:, n])
                S += np.trace(self.sigma_mu[i]) + np.dot(self.means_mu[i].T, self.means_mu[i])
            L = L * S

        # sum ln N(z_n)
        L += - self.N / 2 * self.m * np.log(2*np.pi)
        for n in range(0, self.N):
            L += - 1/2 * (np.trace(self.sigma_z) + np.dot(self.means_z[n].T, self.means_z[n]))

        # sum ln N(W|a)
        for i in range(0, self.d.shape[1]):
            L += - 1/2 * self.m * self.d[i] * np.log(2*np.pi)
            for j in range(0, self.m):
                L += - 1/2 * (self.d[i] * (digamma(self.a_new[i]) - np.log(self.b_new[i][j])) +
                              (self.a_new[i] / self.b_new[i][j]) * (np.trace(self.sigma_w[i]) \
                                                        + np.dot(self.means_w[i].T, self.means_w[i])))

        # sum ln Ga(a_i)
        for i in range(0, self.d.shape[1]):
            L += self.m * (-np.log(gamma(self.a[i])) + self.a[i] * np.log(self.b[i]))
            for j in range(0, self.m):
                L += -(self.a_new[i] - 1) * (np.log(self.b_new[i][j]) + digamma(self.a_new[i])) - \
                    self.b[i] * (self.a_new[i] / self.b[i][j])

        # ln(N(\mu))
        for i in range(0, self.d.shape[1]):
            L += -self.d[i]/2 * np.log(2 * np.pi) + 1/2 * np.log(self.beta[i]) - \
                self.beta[i]/2 * (np.trace(self.sigma_mu[i]) +
                                  np.dot(self.means_mu[i].T, self.means_mu[i]))

        # ln(Wi(\phi))
        for i in range(0, self.d.shape[1]):
            L += multigammaln(self.nu[i]/2,self.d[i]) + self.d[i] * np.log(2) + \
                np.log(np.linalg.det(np.linalg.inv(self.K[i])))

        # Terms from entropies
        # H[Q(Z)]
        L += self.N * (self.m/2 * (1 + np.log(2*np.pi)) + 1/2 * np.log(np.linalg.det(self.sigma_z)))

        # H[Q(\mu)]
        for i in range(0, self.d.shape[1]):
            L += self.d[i]/2 * (1 + np.log(2*np.pi)) + 1/2 * np.log(np.linalg.det(self.sigma_mu[i]))                

        # H[Q(W)]
        for i in range(0, self.d.shape[1]):
            L += self.d[i] * (self.d[i]/2 * (1 + np.log(2*np.pi)) +
                    1/2 * np.log(np.linalg.det(self.sigma_w[i])))

        # H[Q(\alpha)]
        for i in range(0, self.d.shape[1]):
            L += self.m * (self.a_new[i] + np.log(gamma(self.a_new[i])) +
                    (1 - self.a_new[i]) * digamma(self.a_new[i]))
            for j in range(0, self.d[i]):
                L += -np.log(self.b_new[i][j, 0])

        # H[Q(\phi)]
        for i in range(0, self.d.shape[1]):
            L += (self.d[i] + 1)/2 * np.log(np.linalg.det(np.linalg.inv(self.K_tilde[i]))) \
                + self.d[i]/2 * (self.d[i] + 1) * np.log(2) + np.log(gamma(self.nu_tilde[i]/2)) \
                    - 1/2 * (self.nu_tilde[i] - self.d[i] - 1) * \
                        multigammaln(self.nu[i]/2,self.d[i]) + (self.nu_tilde[i] * self.d[i])/2

        return L

    def fit(self, X, iterations=1000, threshold=1):
        self.update_phi(X)
        L_previous = -1000000000
        L_new = 0
        i = 0
        for i in range(iterations):
            self.update_phi(X)
            self.update_mu(X)
            self.update_alpha()
            self.update_w(X)
            self.update_z(X)
            if i % 100 == 1:
                print("Iterations: %d", i)
                print("Lower Bound Value : %d", self.L(X))
            i += 1
            L_previous = L_new
            L_new = self.L(X)
            if abs(L_new - L_previous) < threshold:
                break
