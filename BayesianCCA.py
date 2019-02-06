import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma, wishart


class BayesianCCA(object):

    def __init__(self, d, N, a, b, beta, K, nu):

        self.d = np.zeros(1, d.shape[1])
        for i in range(0, d.shape[1]):
            self.d[i] = d[i]  # number of dimensions of data sources
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
        self.mean_mu = [[] for _ in range(d.shape[1])]
        self.sigma_mu = [[] for _ in range(d.shape[1])]
        self.means_w = [[] for _ in range(d.shape[1])]
        self.sigma_w = [[] for _ in range(d.shape[1])]
        self.a_new = [[] for _ in range(d.shape[1])]
        self.b_new = [[] for _ in range(d.shape[1])]
        self.K_tilde = [[] for _ in range(d.shape[1])]
        self.nu_tilde = [[] for _ in range(d.shape[1])]
        for i in range(0, d.shape[1]):
            self.mean_mu[i] = np.random.randn(d[i], 1)
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
                                     np.trace(np.dot(self.E_phi[0], self.sigma_w[0])) + self.means_w[0].T.dot(self.E_phi[0]).dot(self.means_w[0]) +
                                     np.trace(np.dot(self.E_phi[1], self.sigma_w[0])) + self.means_w[1].T.dot(self.E_phi[1]).dot(self.means_w[1]))
        self.means_z = self.sigma_z.dot(((X[0] - self.mean_mu[0]).T.dot(self.E_phi[0]).dot(self.means_w[0]) +
                                         (X[1] - self.mean_mu[1]).T.dot(self.E_phi[1]).dot(self.means_w[1])).T)

    def update_mu(self, X):
        for i in range(0, self.d.shape[1]):
            self.sigma_mu[i] = np.linalg.inv(
                self.beta[i] * np.identity(self.d[i]) + self.N * self.E_phi[i])
            S = 0
            for n in range(0, self.N):
                x_n = np.reshape(X[i][n].T, (self.d[i], 1))
                z_n = np.reshape(self.means_z[n].T, (self.m, 1))
                S += x_n - np.dot(self.means_w[i], z_n)
            self.mean_mu[i] = self.E_phi[i].dot(self.sigma_mu[i], S)

    def update_w(self, X):
        for i in range(0, self.d.shape[1]):
            E_zz = self.sigma_z[i] + np.dot(self.means_z[i], self.means_z[i].T)
            self.sigma_w[i] = np.linalg.inv(np.diagonal(
                self.a_new[i]/self.b_new[i]) + self.E_phi[i][self.d[i], self.d[i]] * E_zz)
            for n in range(0, self.N):
                S1 = 0
                S2 = 0
                for j in range(0, self.d[i]):
                    if j <= self.m:
                        x_n_j = X[i][n,j].T
                        z_n = np.reshape(self.means_z[n].T, (self.m, 1))
                        S1 += (x_n_j - self.mean_mu[i]
                               ).dot(self.E_phi[i][:, j]).dot(z_n.T)
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
            E_W = self.means_w[i]
            for n in range(0, self.N):
                x_n = np.reshape(X[i][n,:].T, (self.d[i], 1))
                z_n = np.reshape(self.means_z.T[n], (self.m, 1))
                A += np.dot(x_n.T, x_n) + np.trace(self.sigma_mu[i]) + np.dot(
                    self.mean_mu[i].T, self.mean_mu[i])
                A += np.trace(np.dot(np.trace(self.sigma_w[i]) + np.dot(
                    E_W.T, E_W), self.sigma_z + np.dot(z_n, z_n.T)))
                A += 2 * np.dot(np.dot(self.mean_mu[i].T, self.means_w[i]), z_n)
                A += -2 * \
                    np.dot(
                        np.dot(z_n.T, self.means_w[i]), z_n) - 2 * np.dot(z_n.T, self.mean_mu[i])
            self.K_tilde[i] = self.K[i] + 0.5*A

    def L(self, X):

        L = 0
        ###Terms from expectations###
        # N(X_n|Z_n)
        for i in range(0, self.d.shape[1]):
            S = 0
            L += -self.N * self.d[i]/2 * (wishart(self.nu[i],self.K[i]) - np.log((self.nu[i]-self.d[i]-1) / 2) - np.log(2 * np.pi))
            for n in range(0, self.N):
                S += np.dot(X[i][:, n].T, X[i][:, n])
                S -= np.dot(np.dot(X[i][:, n].T, self.means_w[i]), self.means_z[:, n])
                S += np.dot(X[i][:, n].T, self.mean_mu[i])[0]
                S -= np.dot(np.dot(self.means_w[i].T, self.means_z[:, n].T), X[i][:, n])
                S -= np.dot(np.dot(self.means_w[i].T, self.means_z[:, n].T), self.mean_mu[i])
                S += np.dot(self.mean_mu[i].T, X[i][:, n])[0]
                S -= np.dot(np.dot(self.mean_mu[i].T, self.means_w[i]), self.means_z[:, n])
                S += np.trace(self.sigma_mu[i]) + np.dot(self.mean_mu[i].T, self.mean_mu[i])[0]
                S += np.dot(self.mean_mu[i].T, X[i][:, n].T)[0]
                S -= np.dot(np.dot(self.mean_mu[i].T, self.means_w[i]) , self.means_z[:, n])[0]
                S += np.trace(self.sigma_mu[i]) + np.dot(self.mean_mu[i].T, self.mean_mu[i])[0][0]
            L += self.N/2 * np.dot(self.E_phi[i]) 

        # sum ln N(z_n)
        L += - self.N / 2 * (np.trace(self.sigma_z) - self.m * np.log(2*np.pi))
        for n in range(0, self.N):
            L += - 1/2 * np.dot(self.means_z.T[n].T, self.means_z[n])

        # sum ln(W|a)
        L += self.q * self.d / 2 * (- np.log(2*np.pi))
        for i in range(0, self.q):
            L += -self.q / 2 * ((special.digamma(self.a_alpha_tilde) - log(self.bs_alpha_tilde[i][0])) + self.a_alpha_tilde / self.bs_alpha_tilde[i][0]) / (- 1/2) * (
                np.trace(self.sigma_w) + np.dot(self.means_w.T[i].T, self.means_w.T[i]))

        # sum ln (Ga(a_i))
        L += self.q * (- log(special.gamma(self.a_alpha)) +
                       self.a_alpha * log(self.b_alpha))
        for i in range(0, self.q):
            L += -log(self.bs_alpha_tilde[i][0]) + special.digamma(
                self.a_alpha_tilde) - self.b_alpha * (self.a_alpha_tilde / self.bs_alpha_tilde[i][0])

        # ln(N(\mu))
        for i in range(0, self.d.shape[1]):
            L += self.d[i]/2 * (np.log(self.beta[i]) - np.log(2 * np.pi)) - self.beta[i]/2 * \
                (np.trace(self.sigma_mu[i]) + np.dot(self.mean_mu[i].T, self.mean_mu[i])[0][0])

        # ln(Wi(\phi))
        L += -log(self.b_tau_tilde[0]) + special.digamma(
            self.a_tau_tilde) - (self.a_tau_tilde / self.b_tau_tilde[0])

        # Terms from entropies
        # H[Q(Z)]
        L += self.N*(self.d/2*(1 + np.log(2*np.pi)) +
                     1/2 * log(linalg.det(self.sigma_z)))

        # H[Q(\mu)]
        L += (0.5)*log(linalg.det(self.sigma_mu)) + \
            self.d/2*(1 + np.log(2*np.pi))

        # H[Q(W)]
        L += self.d*(self.d / 2 * (1 + np.log(2*np.pi)) +
                     1/2*log(linalg.det(self.sigma_w)))

        # H[Q(\alpha)]
        L += self.q * (self.a_alpha_tilde + log(special.gamma(self.a_alpha_tilde)
                                                ) + (1-self.a_alpha_tilde)*special.digamma(self.a_alpha_tilde))
        for i in range(0, self.d):
            L += -log(self.bs_alpha_tilde[i][0])

        # H[Q(\tau)]
        L += self.a_tau_tilde - \
            log(self.b_tau_tilde[0]) + (1-self.a_tau_tilde) * \
            special.digamma(self.a_tau_tilde)
        # the term Gamma(a_tau_tilde) is inf so we ignore it...

        return L

    def fit(self, X, iterations=1000, threshold=1):
        self.update_tau(X)
        L_previous = -1000000000
        L_new = 0
        i = 0
        for i in range(iterations):
            self.update_tau(X)
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
