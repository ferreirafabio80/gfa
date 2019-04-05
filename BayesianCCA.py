import numpy as np
import matplotlib.pyplot as plt
from numpy.random import gamma
from scipy.special import multigammaln, digamma


class VCCA(object):

    def __init__(self, X, m, d):


        self.d = d  # number of dimensions of data sources
        self.m = m   # number of different models
        self.N = np.shape(X[0])[0]  # number of data points

        ## Hyperparameters
        self.a = self.b = self.beta = np.array([10e-03, 10e-03])
        self.K = np.array([10e-03 * np.identity(d[0]),10e-03 * np.identity(d[1])])
        self.nu = np.array([d[0] + 1, d[1] + 1])

        ## Initialising variational parameters
        # Latent variables
        self.means_z = np.random.randn(self.N, m)
        self.sigma_z = np.identity(m)
        # Means
        self.means_mu = [[] for _ in range(d.size)]
        self.sigma_mu = [[] for _ in range(d.size)]
        # Projection matrices
        self.means_w = [[] for _ in range(d.size)]
        self.sigma_w = [[] for _ in range(d.size)]
        # ARD parameters (Gamma distribution)
        #-the parameters for the ARD precisions
        self.a_ard = [[] for _ in range(d.size)]
        self.b_ard = [[] for _ in range(d.size)]
        #-the mean of the ARD precisions
        self.E_alpha = [[] for _ in range(d.size)]
        # Precisions (Wishart distribution)
        #-degrees of freedom
        self.nu_tilde = [[] for _ in range(d.size)]
        #-scale matrix
        self.K_tilde = [[] for _ in range(d.size)]
        # Data variance needed for sacling alphas
        self.datavar = [[] for _ in range(d.size)]
        for i in range(0, d.size):
            self.means_mu[i] = np.random.randn(d[i], 1)
            self.sigma_mu[i] = np.identity(d[i])
            self.means_w[i] = np.random.randn(d[i], m)
            self.sigma_w[i] = np.random.randn(m, m, d[i])
            self.a_ard[i] = self.a[i] + d[i]/2.0
            self.b_ard[i] = np.ones((1, self.m)))
            self.nu_tilde[i] = self.nu[i] + self.N
            self.K_tilde[i] = np.identity(d[i])
            self.datavar[i] = np.sum(X[i].var(0))
            self.E_phi[i] = self.nu_tilde[i] * self.K_tilde[i]
            self.E_alpha[i] = m * d[i]/(self.datavar[i] - \
                1/np.diag(self.E_phi[i])) 

    def update_w(self, X):
        for i in range(0, self.d.size): 
            ## Update covariance matrices of Ws         
            E_zz = self.N * self.sigma_z + np.dot(self.means_z, self.means_z.T)
            for j in range(0, self.d[i]):
                self.sigma_w[i][:,:,j] = np.linalg.inv(np.diagflat(self.E_alpha) + \
                    self.E_phi[i][j,j] * E_zz)

            ## Update expectations of Ws  
            l_vals = np.array([range(0, self.d[i])])
            for j in range(0, self.d[i]):
                S = 0
                l_new = np.delete(l_vals, j)
                for n in range(0, self.N):
                    #first term of the sum
                    x_n = np.reshape(X[i][n, :], (self.d[i], 1))
                    z_n = np.reshape(self.means_z[:, n], (self.m,1))
                    E_phi_j  = np.reshape(self.E_phi[i][:, j], (self.d[i],1))
                    A1 = np.dot((x_n - self.means_mu[i]).T, np.dot(E_phi_j,z_n.T))                  
                    
                    #second term of the sum
                    zz_n = self.sigma_z + np.dot(z_n, z_n.T)
                    A2 = 0
                    for l in l_new:  
                        EW_l = np.reshape(self.means_w[i][l, :], (self.m, 1))
                        A2 += EW_l * self.E_phi[i][l,j]
                    #combining both terms
                    S += A1 - np.dot(zz_n, A2).T    
                self.means_w[i][j,:]= np.reshape(np.dot(
                            self.sigma_w[i][:,:,j], S.T),(1,self.m))

    def update_z(self, X):
        ## Update covariance matrix of Z
        E_WphiW = np.zeros((self.m, self.m))
        for i in range(0, self.d.size):
            E_phiDiagCovW = np.zeros((self.m, self.m))
            for j in range(0, self.d[i]):
                E_phiDiagCovW += self.E_phi[i][j,j] * self.sigma_w[i][:,:,j] 
            E_W = self.means_w[i]
            E_WphiW += np.dot(E_W.T, self.E_phi[i]).dot(E_W) + E_phiDiagCovW
        self.sigma_z = np.linalg.inv(np.identity(self.m) + E_WphiW)
        
        ## Update expectations of Z
        S = 0 
        for i in range(0, self.d.size):
            A = X[i].T - self.means_mu[i]
            S += np.dot(A.T, self.E_phi[i]).dot(self.means_w[i]) 
        self.means_z = np.dot(self.sigma_z, S.T)

    def update_mu(self, X):
        self.E_phi = [[] for _ in range(self.d.size)]
        for i in range(0, self.d.size):
            self.E_phi[i] = self.nu_tilde[i] * np.linalg.inv(self.K_tilde[i])
            
            ## Update covariance matrix of mus 
            self.sigma_mu[i] = np.linalg.inv(self.beta[i] * np.identity(self.d[i]) +
                self.N * self.E_phi[i])
            
            ## Update expectations of mus
            S = 0
            for n in range(0, self.N):
                x_n = np.reshape(X[i][n, :], (self.d[i], 1))
                z_n = np.reshape(self.means_z[:, n], (self.m, 1))
                S += x_n - np.dot(self.means_w[i], z_n)
            self.means_mu[i] = np.dot(self.sigma_mu[i],self.E_phi[i]).dot(S)

    def update_alpha(self):
        for i in range(0, self.d.size):         
            ## Update b
            for k in range(0, self.m):    
                self.b_ard[i][k] = self.b[i] + 0.5 * (np.trace(self.covW[i]) + \
                    np.dot(self.means_w[i][:,k].T,self.means_w[i][:,k]))
            self.E_alpha[i] = self.a_ard[i] / self.b_ard[i]        

    def update_phi(self, X):
        for i in range(0, self.d.size):         
            ## Update K 
            S = 0
            for n in range(0, self.N):
                x_n = np.reshape(X[i][n, :], (self.d[i], 1))
                z_n = np.reshape(self.means_z[:, n], (self.m, 1))
                S += np.dot(x_n, x_n.T)
                S -= np.dot(x_n, np.dot(z_n.T, self.means_w[i].T))
                S -= np.dot(x_n, self.means_mu[i].T)
                S -= np.dot(self.means_w[i], z_n).dot(x_n.T)
                S += np.dot(self.means_w[i], z_n).dot(self.means_mu[i].T)
                S -= np.dot(self.means_mu[i], x_n.T)
                S += np.dot(self.means_mu[i], np.dot(z_n.T, self.means_w[i].T))
                S += self.sigma_mu[i] + np.dot(self.means_mu[i], self.means_mu[i].T)
                S += np.trace(self.d[i] * np.trace(self.covW[i]) + np.dot(self.means_w[i].T, \
                    self.means_w[i]) * (self.sigma_z + np.dot(self.means_z, self.means_z.T)))
            self.K_tilde[i] = self.K[i] + S 

    def L(self, X):
        ## Compute the lower bound##
        
        ## Terms from expectations        
        # N(X_n|Z_n)
        L = 0
        for i in range(0, self.d.size):
            symmPSD_K = np.dot(self.K_tilde[i],self.K_tilde[i].T)
            chol_logphi = np.log(np.diagonal(np.linalg.cholesky(symmPSD_K)))  
            L += -self.N/2 * (self.d[i] * np.log(2*np.pi) - multigammaln(self.nu_tilde[i]/2,self.d[i]) - \
                self.d[i] * np.log(2) - (-2 * np.sum(chol_logphi)))                     
            S = 0
            for n in range(0, self.N):
                x_n = np.reshape(X[i][n, :], (self.d[i], 1))
                z_n = np.reshape(self.means_z[:, n], (self.m, 1))
                S += np.dot(x_n.T, x_n)
                S -= np.dot(x_n.T, self.means_w[i]).dot(z_n)
                S -= np.dot(x_n.T, self.means_mu[i])
                S -= np.dot(z_n.T, self.means_w[i].T).dot(x_n)
                S += np.dot(z_n.T, self.means_w[i].T).dot(self.means_mu[i])
                S -= np.dot(self.means_mu[i].T, x_n)
                S += np.dot(self.means_mu[i].T, np.dot(self.means_w[i], z_n))
                S += np.trace(self.sigma_mu[i]) + np.dot(self.means_mu[i].T, self.means_mu[i])
                S += np.trace(self.d[i] * np.trace(self.covW[i]) + np.dot(self.means_w[i].T, \
                    self.means_w[i]) * (self.sigma_z + np.dot(self.means_z, self.means_z.T))) 
            L -= self.N/2 * self.E_phi[i][0,0] * S # this is wrong 

        # sum ln N(z_n)
        L += - self.N / 2 * self.m * np.log(2*np.pi)
        for n in range(0, self.N):
            z_n = np.reshape(self.means_z[:, n], (self.m, 1))
            L += - 1/2 * (np.trace(self.sigma_z) + np.dot(z_n.T, z_n))

        # sum ln N(W|a)
        for i in range(0, self.d.size):
            L += - 1/2 * self.m * self.d[i] * np.log(2*np.pi)
            for j in range(0, self.m):
                L += - 1/2 * (self.d[i] * (digamma(self.a_ard[i]) - np.log(self.b_ard[i][j])) +
                    (self.a_ard[i] / self.b_ard[i][j]) * (np.trace(self.covW[i]) \
                        + np.dot(self.means_w[i][:,j].T, self.means_w[i][:,j])))

        # sum ln Ga(a_i)
        for i in range(0, self.d.size):
            L += self.m * (-np.log(gamma(self.a[i])) + self.a[i] * np.log(self.b[i]))
            for j in range(0, self.m):
                L += -(self.a[i] - 1) * (np.log(self.b_ard[i][j]) + digamma(self.a_ard[i])) - \
                    self.b[i] * (self.a_ard[i] / self.b_ard[i][j])

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
            L += self.d[i] * (self.d[i]/2 * (1 + np.log(2*np.pi))) 
            for j in range(0, self.d[i]):
                L += 1/2 * np.log(np.linalg.det(self.covW[i]))

        # H[Q(\alpha)]
        for i in range(0, self.d.size):
            L += self.m * (self.a_ard[i] + np.log(gamma(self.a_ard[i])) +
                    (1 - self.a_ard[i]) * digamma(self.a_ard[i]))
            for j in range(0, self.m):
                L += -np.log(self.b_ard[i][j])

        # H[Q(\phi)]
        for i in range(0, self.d.size):
            symmPSD_K = np.dot(self.K_tilde[i],self.K_tilde[i].T)
            chol_logphi = np.log(np.diagonal(np.linalg.cholesky(symmPSD_K)))
            L += (self.d[i] + 1)/2 * (-2 * np.sum(chol_logphi)) \
                + self.d[i]/2 * (self.d[i] + 1) * np.log(2) + multigammaln(self.nu[i]/2,self.d[i]) \
                    - 1/2 * (self.nu_tilde[i] - self.d[i] - 1) * digamma(multigammaln(self.nu[i]/2, self.d[i])) \
                        + (self.nu_tilde[i] * self.d[i])/2

        return L

    def fit(self, X, iterations=10000, threshold=10e-06):
        L_previous = 0
        L_mat = []
        for i in range(iterations):
            self.update_w(X)
            self.update_z(X) 
            self.update_alpha()
            self.update_phi(X) 
            self.update_mu(X)                 
            print("Iterations: %d", i+1)
            print("Lower Bound Value : %d", self.L(X))
            L_new = self.L(X)
            L_mat.append(L_new)
            if abs(L_new - L_previous) < threshold:
                break
            L_previous = L_new
        return L_mat
