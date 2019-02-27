import numpy as np
import matplotlib.pyplot as plt
from numpy.random import gamma
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
        self.means_mu = [[] for _ in range(d.size)]
        self.sigma_mu = [[] for _ in range(d.size)]
        self.means_w = [[] for _ in range(d.size)]
        self.sigma_w = [[] for _ in range(d.size)]
        self.a_new = [[] for _ in range(d.size)]
        self.b_new = [[] for _ in range(d.size)]
        self.K_tilde = [[] for _ in range(d.size)]
        self.nu_tilde = [[] for _ in range(d.size)]
        for i in range(0, d.size):
            self.means_mu[i] = np.random.randn(d[i], 1)
            self.sigma_mu[i] = np.random.randn(d[i], d[i])
            self.means_w[i] = np.random.randn(d[i], m)
            self.sigma_w[i] = np.random.randn(m, m)
            self.a_new[i] = self.a[i] + d[i]/2.0
            self.b_new[i] = np.abs(np.random.randn(m, 1))
            self.K_tilde[i] = np.abs(np.random.randn(d[i], d[i]))
            self.nu_tilde[i] = self.nu[i] + N

    def update_z(self, X):
        #E_WW1 = np.trace(self.sigma_w[0]) + np.dot(self.means_w[0].T, self.means_w[0])
        #E_WW2 = np.trace(self.sigma_w[1]) + np.dot(self.means_w[1].T, self.means_w[1]) 
        self.sigma_z = np.linalg.inv(np.identity(self.m) + np.dot(np.dot(self.means_w[0].T, 
            self.E_phi[0]), np.dot(self.E_phi[0],self.means_w[0])) + np.dot(np.dot(
                self.means_w[1].T, self.E_phi[1]), np.dot(self.E_phi[1],self.means_w[1])))
        S = 0 
        for i in range(0, self.d.size):
            A = np.zeros((self.d[i], self.N))
            for n in range(0, self.N):
                x_n = np.reshape(X[i][n, :], (self.d[i], 1))
                A[:, [n]] = x_n - self.means_mu[i]
            S += np.dot(A.T, self.E_phi[i]).dot(self.means_w[i]) 
        self.means_z = np.dot(self.sigma_z, S.T)

    def update_mu(self, X):
        self.E_phi = [[] for _ in range(self.d.size)]
        for i in range(0, self.d.size):
            self.E_phi[i] = self.nu_tilde[i] * np.linalg.inv(self.K_tilde[i])
            
            self.sigma_mu[i] = np.linalg.inv(self.beta[i] * np.identity(self.d[i]) +
                self.N * self.E_phi[i])
            S = 0
            for n in range(0, self.N):
                x_n = np.reshape(X[i][n, :], (self.d[i], 1))
                z_n = np.reshape(self.means_z[:, n], (self.m, 1))
                S += x_n - np.dot(self.means_w[i], z_n)
            self.means_mu[i] = np.dot(self.sigma_mu[i],self.E_phi[i]).dot(S)

    def update_w(self, X):
        for i in range(0, self.d.size):
            E_zz = self.N * np.trace(self.sigma_z) + np.dot(self.means_z, self.means_z.T)
            A = 0
            for j in range(0, self.d[i]):
                A += self.E_phi[i][j,j] * E_zz
            self.sigma_w[i] = np.linalg.inv(np.diagflat(self.a_new[i]/self.b_new[i]) + A)
            
            l = np.array([range(0, self.d[i])]) 
            for j in range(0, self.d[i]):
                S = np.zeros((1,self.m))
                for n in range(0, self.N):
                    x_n_j = X[i][n, j]
                    z_n = np.reshape(self.means_z[:, n], (self.m, 1))
                    E_phi_j  = np.reshape(self.E_phi[i][:, j], (self.d[i], 1))
                    S += (x_n_j - self.means_mu[i][j]) * np.dot(E_phi_j, z_n.T)[j,:]                   
                    
                    l_new = np.delete(l, j)
                    for k in l_new:
                        zz_n = np.trace(self.sigma_z) + np.dot(z_n, z_n.T)
                        EW_n_k = np.reshape(self.means_w[i][k, :], (self.m, 1))
                        S -= (np.dot(zz_n, EW_n_k) * self.E_phi[i][k,j]).T
            self.means_w[i][j,:] = np.reshape(np.dot(self.sigma_w[i], S.T), (1, self.m))

    def update_alpha(self):
        for i in range(0, self.d.size):
            for j in range(0, self.m):
                EW_j = np.reshape(self.means_w[i][:, j], (self.d[i], 1))
                self.b_new[i][j] = self.b[i] + (np.trace(self.sigma_w[i]) + np.dot(EW_j.T, EW_j)) / 2

    def update_phi(self, X):
        for i in range(0, self.d.size):
            A = 0
            for n in range(0, self.N):
                x_n = np.reshape(X[i][n, :], (self.d[i], 1))
                z_n = np.reshape(self.means_z[:, n], (self.m, 1))
                A += np.dot(x_n.T, x_n)
                A -= np.dot(x_n.T, self.means_w[i]).dot(z_n)
                A -= np.dot(x_n.T, self.means_mu[i])
                A -= np.dot(self.means_w[i], z_n).T.dot(x_n)
                A += np.trace(np.trace(self.sigma_w[i]) + np.dot(self.means_w[i].T, 
                    self.means_w[i]) * (np.trace(self.sigma_z) + np.dot(z_n.T,z_n)))
                A += np.dot(self.means_w[i], z_n).T.dot(self.means_mu[i])
                A -= np.dot(self.means_mu[i].T, x_n)
                A += np.dot(self.means_mu[i].T, self.means_w[i]).dot(z_n)
                A += np.trace(self.sigma_mu[i]) + np.dot(self.means_mu[i].T, self.means_mu[i])
            self.K_tilde[i] = self.K[i] + A * np.identity(self.d[i])

    def L(self, X):

        L = 0
        ###Terms from expectations###
        # N(X_n|Z_n)
        for i in range(0, self.d.size):
            S = 0
            L += -self.N/2 * (self.d[i] * np.log(2*np.pi) - multigammaln(self.nu_tilde[i]/2,self.d[i]) - \
                self.d[i] * np.log(2) - np.log(np.linalg.det(np.linalg.inv(self.K_tilde[i])))) + \
                    (self.nu_tilde[i] * self.K_tilde[i])[0][0]
            for n in range(0, self.N):
                x_n = np.reshape(X[i][n, :], (self.d[i], 1))
                z_n = np.reshape(self.means_z[:, n], (self.m, 1))
                S += np.dot(x_n.T, x_n)
                S -= np.dot(x_n.T, self.means_w[i]).dot(z_n)
                S -= np.dot(x_n.T, self.means_mu[i])
                S -= np.dot(self.means_w[i], z_n).T.dot(x_n)
                S += np.dot(self.means_w[i], z_n).T.dot(self.means_w[i]).dot(z_n)
                S += np.trace(np.trace(self.sigma_w[i]) + np.dot(self.means_w[i].T, 
                    self.means_w[i]) * (np.trace(self.sigma_z) + np.dot(z_n.T,z_n)))
                S += np.dot(self.means_w[i], z_n).T.dot(self.means_mu[i])
                S -= np.dot(self.means_mu[i].T, x_n)
                S += np.dot(self.means_mu[i].T, self.means_w[i]).dot(z_n)
                S += np.trace(self.sigma_mu[i]) + np.dot(self.means_mu[i].T, self.means_mu[i])
            L = L * S

        # sum ln N(z_n)
        L += - self.N / 2 * self.m * np.log(2*np.pi)
        for n in range(0, self.N):
            z_n = np.reshape(self.means_z[:, n], (self.m, 1))
            L += - 1/2 * (np.trace(self.sigma_z) + np.dot(z_n.T, z_n))

        # sum ln N(W|a)
        for i in range(0, self.d.size):
            L += - 1/2 * self.m * self.d[i] * np.log(2*np.pi)
            for j in range(0, self.m):
                L += - 1/2 * (self.d[i] * (digamma(self.a_new[i]) - np.log(self.b_new[i][j])) +
                              (self.a_new[i] / self.b_new[i][j]) * (np.trace(self.sigma_w[i]) \
                                                        + np.dot(self.means_w[i][:,j].T, self.means_w[i][:,j])))

        # sum ln Ga(a_i)
        for i in range(0, self.d.size):
            L += self.m * (-np.log(gamma(self.a[i])) + self.a[i] * np.log(self.b[i]))
            for j in range(0, self.m):
                L += -(self.a_new[i] - 1) * (np.log(self.b_new[i][j]) + digamma(self.a_new[i])) - \
                    self.b[i] * (self.a_new[i] / self.b_new[i][j])

        # ln(N(\mu))
        for i in range(0, self.d.size):
            L += -self.d[i]/2 * np.log(2 * np.pi) + 1/2 * np.log(self.beta[i]) - \
                self.beta[i]/2 * (np.trace(self.sigma_mu[i]) +
                                  np.dot(self.means_mu[i].T, self.means_mu[i]))

        # ln(Wi(\phi))
        for i in range(0, self.d.size):
            L += multigammaln(self.nu[i]/2,self.d[i]) + self.d[i] * np.log(2) + \
                np.log(np.linalg.det(np.linalg.inv(self.K[i])))

        # Terms from entropies
        # H[Q(Z)]
        L += self.N * (self.m/2 * (1 + np.log(2*np.pi)) + 1/2 * np.log(np.linalg.det(self.sigma_z)))

        # H[Q(\mu)]
        for i in range(0, self.d.size):
            L += self.d[i]/2 * (1 + np.log(2*np.pi)) + 1/2 * np.log(np.linalg.det(self.sigma_mu[i]))                

        # H[Q(W)]
        for i in range(0, self.d.size):
            L += self.d[i] * (self.d[i]/2 * (1 + np.log(2*np.pi)) +
                    1/2 * np.log(np.linalg.det(self.sigma_w[i])))

        # H[Q(\alpha)]
        for i in range(0, self.d.size):
            L += self.m * (self.a_new[i] + np.log(gamma(self.a_new[i])) +
                    (1 - self.a_new[i]) * digamma(self.a_new[i]))
            for j in range(0, self.m):
                L += -np.log(self.b_new[i][j])

        # H[Q(\phi)]
        for i in range(0, self.d.size):
            L += (self.d[i] + 1)/2 * np.log(np.linalg.det(np.linalg.inv(self.K_tilde[i]))) \
                + self.d[i]/2 * (self.d[i] + 1) * np.log(2) + np.log(gamma(self.nu_tilde[i]/2)) \
                    - 1/2 * (self.nu_tilde[i] - self.d[i] - 1) * \
                        multigammaln(self.nu[i]/2,self.d[i]) + (self.nu_tilde[i] * self.d[i])/2

        return L

    def fit(self, X, iterations=100, threshold=1):
        L_previous = -10000000
        L_mat = []
        for i in range(iterations):
            self.update_phi(X)
            self.update_mu(X)
            self.update_alpha()
            self.update_w(X)
            self.update_z(X)
            if i % 10 == 1:
                print("Iterations: %d", i)
                print("Lower Bound Value : %d", self.L(X))
            L_new = self.L(X)
            L_mat[i] = L_new
            if abs(L_new - L_previous) < threshold:
                break
            L_previous = L_new
            i += 1
        return L_mat
