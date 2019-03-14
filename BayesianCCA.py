import numpy as np
import matplotlib.pyplot as plt
from numpy.random import gamma
from numpy.matlib import repmat
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
            self.sigma_w[i] = np.random.randn(m, m, d[i])
            self.a_new[i] = self.a[i] + d[i]/2.0
            self.b_new[i] = np.abs(np.random.randn(self.m, 1))
            self.K_tilde[i] = np.abs(np.random.randn(d[i], d[i]))
            self.nu_tilde[i] = self.nu[i] + N

    def update_z(self, X):
        A1 = np.zeros((self.m, self.m))
        for i in range(0, self.d.size):
            A2 = np.zeros((self.m, self.m))
            for j in range(0, self.d[i]):
                A2 += self.E_phi[i][j,j] * self.sigma_w[i][:,:,j] 
            E_W = self.means_w[i]
            A1 += np.dot(E_W.T, self.E_phi[i]).dot(E_W) + A2
        self.sigma_z = np.linalg.inv(np.identity(self.m) + A1)
        
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
            E_zz = self.N * self.sigma_z + np.dot(self.means_z, self.means_z.T)
            for j in range(0, self.d[i]):
                self.alpha = self.a_new[i]/self.b_new[i]
                self.sigma_w[i][:,:,j] = np.linalg.inv(np.diagflat(self.alpha) + \
                    self.E_phi[i][j,j] * E_zz)
            
            l = np.array([range(0, self.d[i])])
            for n in range(0, self.N):
                S = 0
                for j in range(0, self.d[i]):
                    l_new = np.delete(l, j)
                    x_n_j = X[i][n, j]
                    z_n = np.reshape(self.means_z[:, n], (self.m, 1))
                    zz_n = self.sigma_z + np.dot(z_n, z_n.T)
                    E_phi_j  = np.reshape(self.E_phi[i][:, j], (self.d[i],1))
                    A1 = (x_n_j - self.means_mu[i][j]) * np.dot(E_phi_j,z_n.T)                  
                    A2 = 0
                    for k in l_new:  
                        EW_n_k = np.reshape(self.means_w[i][k, :], (1, self.m))
                        A2 += EW_n_k.T * self.E_phi[i][k,j]
                    S += np.reshape(A1, (self.m, self.d[i])) - np.dot(zz_n, A2)    
            self.means_w[i]= np.reshape(np.dot(self.sigma_w[i][:,:,j], S),(self.d[i], self.m))

    def update_alpha(self):
        for i in range(0, self.d.size):
            varW = np.zeros((self.d[i], self.m))
            for j in range(0, self.d[i]):
                varW[j, :] = np.reshape(np.diag(self.sigma_w[i][:,:,j]), (1, self.m))

            self.b_new[i] = self.b[i] + 0.5 * \
                 (self.means_w[i] * self.means_w[i] + varW).sum(axis=0)

    def update_phi(self, X):
        for i in range(0, self.d.size):
            E_zz = self.N * self.sigma_z + np.dot(self.means_z, self.means_z.T)
            diag = np.zeros((self.d[i], self.d[i]))
            for j in range(0, self.d[i]):
                diag[j,j] += np.trace(np.dot(self.sigma_w[i][:,:,j], E_zz))   
            S = 0
            for n in range(0, self.N):
                x_n = np.reshape(X[i][n, :], (self.d[i], 1))
                z_n = np.reshape(self.means_z[:, n], (self.m, 1))
                S += np.dot(x_n, x_n.T)
                S -= np.dot(x_n, np.dot(z_n.T, self.means_w[i].T))
                S -= np.dot(x_n, self.means_mu[i].T)
                S -= np.dot(self.means_w[i], z_n).dot(x_n.T)
                S += diag
                S += np.dot(self.means_w[i], z_n).dot(self.means_mu[i].T)
                S -= np.dot(self.means_mu[i], x_n.T)
                S += np.dot(self.means_mu[i], np.dot(z_n.T, self.means_w[i].T))
                S += self.sigma_mu[i] + np.dot(self.means_mu[i], self.means_mu[i].T)
            self.K_tilde[i] = self.K[i] + S 

    def L(self, X):

        ###Terms from expectations###
        # N(X_n|Z_n)
        L = 0
        Ls = [[] for _ in range(self.d.size)]
        for i in range(0, self.d.size):
            E_zz = self.N * self.sigma_z + np.dot(self.means_z, self.means_z.T)
            diag = np.zeros((self.d[i], self.d[i]))
            for j in range(0, self.d[i]):
                diag[j,j] += np.trace(np.dot(self.sigma_w[i][:,:,j], E_zz))      
            
            S = 0
            Ls[i] = -self.N/2 * (self.d[i] * np.log(2*np.pi) - multigammaln(self.nu_tilde[i]/2,self.d[i]) - \
                self.d[i] * np.log(2) - np.log(np.linalg.det(np.linalg.inv(self.K_tilde[i])))) + \
                    (self.nu_tilde[i] * np.linalg.inv(self.K_tilde[i]))
            for n in range(0, self.N):
                x_n = np.reshape(X[i][n, :], (self.d[i], 1))
                z_n = np.reshape(self.means_z[:, n], (self.m, 1))
                S += np.dot(x_n.T, x_n)
                S -= np.dot(x_n.T, self.means_w[i]).dot(z_n)
                S -= np.dot(x_n.T, self.means_mu[i])
                S -= np.dot(z_n.T, self.means_w[i].T).dot(x_n)
                S = S[0] + diag
                S += np.dot(z_n.T, self.means_w[i].T).dot(self.means_mu[i])
                S -= np.dot(self.means_mu[i].T, x_n)
                S += np.dot(self.means_mu[i].T, np.dot(self.means_w[i], z_n))
                S += np.trace(self.sigma_mu[i]) + np.dot(self.means_mu[i].T, self.means_mu[i])
            Ls[i] = Ls[i] * S
            L += Ls[i][0,0]

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
        L += self.N * (self.m/2 * (1 + np.log(2*np.pi)) + 1/2 * np.log(
            np.linalg.det(self.sigma_z)))

        # H[Q(\mu)]
        for i in range(0, self.d.size):
            L += self.d[i]/2 * (1 + np.log(2*np.pi)) + 1/2 * np.log(np.linalg.det(
                self.sigma_mu[i]))                

        # H[Q(W)]
        for i in range(0, self.d.size):
            A = np.zeros((self.m, self.m))
            for j in range(0, self.d[i]):
                A += self.sigma_w[i][:,:,j] 
            L += self.d[i] * (self.d[i]/2 * (1 + np.log(2*np.pi)) +
                    1/2 * np.log(np.linalg.det(A)))

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

    def fit(self, X, iterations=100, threshold=10e-06):
        L_previous = 0
        L_mat = []
        for i in range(iterations):
            self.update_alpha()
            self.update_mu(X)
            self.update_w(X)
            self.update_phi(X)            
            self.update_z(X)   
            #if i % 10 == 1:
            print("Iterations: %d", i+1)
            print("Lower Bound Value : %d", self.L(X))
            L_new = self.L(X)
            L_mat.append(L_new)
            if abs(L_new - L_previous) < threshold:
                break
            L_previous = L_new
            i += 1
        return L_mat
