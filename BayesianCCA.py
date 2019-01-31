import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma, wishart 

class BayesianCCA(object):

    def __init__(self, d, N, a, b, beta, K, nu):

        self.d = np.zeros(1,d.shape[1])
        for i in range(0,d.shape[1]):
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
        self.K_new = [[] for _ in range(d.shape[1])]
        self.nu_new = [[] for _ in range(d.shape[1])]
        for i in range(0,d.shape[1]):
            self.mean_mu[i] = np.random.randn(d[i], 1)
            self.sigma_mu[i] = np.random.randn(d[i], d[i])
            self.means_w[i] = np.random.randn(d[i], m)
            self.sigma_w[i] = np.random.randn(m, m)
            self.a_new[i] = self.a + d[i]/2.0
            self.b_new[i] = np.abs(np.random.randn(m, 1))
            self.K_new[i] = np.abs(np.random.randn(d[i], d[i]))
            self.nu_new[i] = self.nu + N

    def update_z(self, X):
        self.E_phi = [[] for _ in range(self.d.shape[1])]
        for i in range(0,self.d.shape[1]):
            self.E_phi[i] = np.dot(self.nu_new[i],np.linalg.inv(self.K_new[i]))
        
        self.sigma_z = np.linalg.inv(np.identity(self.m) + \
                            np.trace(np.dot(self.E_phi[0],self.sigma_w[0])) + self.means_w[0].T.dot(self.E_phi[0]).dot(self.means_w[0]) + \
                            np.trace(np.dot(self.E_phi[1],self.sigma_w[0])) + self.means_w[1].T.dot(self.E_phi[1]).dot(self.means_w[1]))
        self.means_z = self.sigma_z.dot(((X[0] - self.mean_mu[0]).T.dot(self.E_phi[0]).dot(self.means_w[0]) + \
                                         (X[1] - self.mean_mu[1]).T.dot(self.E_phi[1]).dot(self.means_w[1])).T)

    def update_mu(self,X):
        for i in range(0,self.d.shape[1]):
            self.sigma_mu[i] = np.linalg.inv(self.beta[i] * np.identity(self.d[i]) + self.N * self.E_phi[i])
            S = 0 
            for n in range(0,self.N):
                x_n = np.reshape(X[i].T[n],(self.d[i],1)) 
                z_n = np.reshape(self.means_z.T[n],(self.m,1)) 
                S += x_n - np.dot(self.means_w[i],z_n)
            self.mean_mu[i] = self.E_phi[i].dot(self.sigma_mu[i],S)
                                    