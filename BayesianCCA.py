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
        E_tau = self.a_tau_tilde / self.b_tau_tilde
        E_W = self.means_w
        E_mu = self.mean_mu
        for n in range(0,self.N):
            t_n = np.reshape(X.T[n],(self.d,1)) 
            self.means_z.T[n] = reshape(E_tau * dot ( dot( self.sigma_z , E_W.T ) , (t_n - E_mu)) ,self.q)
        self.sigma_z = np.linalg.inv(np.identity(self.q) + E_tau*(dot(E_W.T,E_W)))