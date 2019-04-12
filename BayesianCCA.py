import numpy as np
import matplotlib.pyplot as plt
from numpy.matlib import repmat
from numpy.random import gamma
from scipy.special import multigammaln, digamma


class VCCA(object):

    def __init__(self, X, m, d):

        self.s = d.size # number of sources
        self.d = d  # dimensions of data sources
        self.m = m   # number of different models
        self.N = X[0].shape[1]  # data points

        ## Hyperparameters
        self.a = self.b = self.beta = np.array([10e-03, 10e-03])
        self.K = np.array([10e-03 * np.identity(d[0]),10e-03 * np.identity(d[1])])
        self.nu = np.array([d[0] + 1, d[1] + 1])

        ## Initialising variational parameters
        # Latent variables
        self.sigma_z = np.identity(m)
        self.means_z = np.random.randn(m, self.N)
        # Means
        self.means_mu = [[] for _ in range(self.s)]
        self.sigma_mu = [[] for _ in range(self.s)]
        # Projection matrices
        self.means_w = [[] for _ in range(self.s)]
        self.sigma_w = [[] for _ in range(self.s)]
        # ARD parameters (Gamma distribution)
        #-the parameters for the ARD precisions
        self.a_ard = [[] for _ in range(self.s)]
        self.b_ard = [[] for _ in range(self.s)]
        #-the mean of the ARD precisions
        self.E_alpha = [[] for _ in range(self.s)]
        # Precisions (Wishart distribution)
        #-degrees of freedom
        self.nu_tilde = [[] for _ in range(self.s)]
        #-scale matrix
        self.K_tilde = [[] for _ in range(self.s)]
        #-the mean of phi
        self.E_phi= [[] for _ in range(self.s)]
        # Data variance needed for sacling alphas
        #self.datavar = [[] for _ in range(s)]
        for i in range(0, self.s):
            self.means_mu[i] = np.random.randn(d[i], 1)
            self.sigma_mu[i] = np.identity(d[i])
            self.means_w[i] = np.random.randn(d[i], m)
            self.sigma_w[i] = np.random.randn(m, m, d[i])
            self.a_ard[i] = self.a[i] + d[i]/2.0
            self.b_ard[i] = np.ones((1, self.m))
            self.nu_tilde[i] = self.nu[i] + self.N
            self.K_tilde[i] = np.identity(d[i])
            #self.datavar[i] = np.sum(X[i].var(0))
            self.E_phi[i] = self.nu_tilde[i] * self.K_tilde[i]
            self.E_alpha[i] = self.a_ard[i] / self.b_ard[i]

    def update_w(self, X):
        for i in range(0, self.s):      
            self.E_zz = self.N * self.sigma_z + np.dot(self.means_z, self.means_z.T)
            for j in range(0, self.d[i]):
                ## Update covariance matrices of Ws  
                self.sigma_w[i][:,:,j] = np.linalg.inv(np.diagflat(self.E_alpha[i]) + \
                    self.E_phi[i][j,j] * self.E_zz)

                #Auxiliary matrix for the last part of the moment
                E_phij = np.reshape(self.E_phi[i][j,:], (self.d[i], 1))
                tmp = self.scale_rows(self.means_w[i].T, E_phij)    
                tmp[j,:] = np.zeros(1,self.d[i])

                #Auxiliary matrix for the sum in the first term
                tmp_sum = np.sum(self.scale_rows(self.means_z, np.dot(
                    (X[i] - repmat(self.means_mu[i], 1, self.N)).T,E_phij)),
                        axis = 0)

                ## Update expectations of Ws  
                self.means_w[i][j,:]= (np.dot(self.sigma_w[i][:,:,j], 
                    np.reshape(tmp_sum, (self.m, 1))) - np.dot(self.E_zz, 
                        np.reshape(np.sum(tmp, axis=0), (self.m, 1)))).T

    def update_z(self, X):
        ## Update covariance matrix of Z
        E_WphiW = np.zeros((self.m, self.m))
        for i in range(0, self.s):
            E_phiDiagCovW = np.zeros((self.m, self.m))
            for j in range(0, self.d[i]):
                E_phiDiagCovW += self.E_phi[i][j,j] * self.sigma_w[i][:,:,j] 
            E_W = self.means_w[i]
            E_WphiW += np.dot(E_W.T, self.E_phi[i]).dot(E_W) + E_phiDiagCovW
        self.sigma_z = np.linalg.inv(np.identity(self.m) + E_WphiW)
        PSD_z = np.dot(self.sigma_z.T, self.sigma_z)/50
        cho = np.linalg.cholesky(PSD_z)
        self.detZ = -2 * np.sum(np.log(np.diag(cho)))
        
        ## Update expectations of Z
        S = 0 
        for i in range(0, self.s):
            S += np.dot((X[i] - repmat(self.means_mu[i], 1, self.N)).T, 
                self.E_phi[i]).dot(self.means_w[i]) 
        self.means_z = np.dot(self.sigma_z, S.T)

    def update_alpha(self):
        for i in range(0, self.s):
            ##Get Variances of W
            VarW = np.zeros((self.d[i], self.m))
            for j in range(0, self.d[i]):
                VarW[j,:] = np.diag(self.sigma_w[i][:,:,j])      

            ## Update b
            Wnorm = VarW + (self.means_w[i] * self.means_w[i])
            self.b_ard[i] = repmat(self.b[i], 1, self.m) + (0.5 * \
                np.reshape(np.sum(Wnorm, axis=0), (self.m, 1))).T
            self.E_alpha[i] = self.a_ard[i] / self.b_ard[i] 


    def update_mu(self, X):
        
        for i in range(0, self.s):          
            ## Update covariance matrix of mus 
            self.sigma_mu[i] = np.linalg.inv(self.beta[i] * np.identity(self.d[i]) +
                (self.N * self.E_phi[i]))
            
            ## Update expectations of mus
            tmp = np.sum((X[i] - np.dot(self.means_w[i], self.means_z)), axis=1)
            self.means_mu[i] = np.dot(self.sigma_mu[i],self.E_phi[i]).dot(
                np.reshape(tmp, (self.d[i], 1)))

    def update_phi(self, X):
        self.E_phi = [[] for _ in range(self.s)]
        for i in range(0, self.s):         
            ## Update K
            # Traces on the diagonals 
            diag = np.zeros((self.d[i], self.d[i]))
            for j in range(0, self.d[i]):
                diag[j,j] = np.trace(self.E_zz * self.sigma_w[i][:,:,j])        

            S = 0
            for n in range(0, self.N):
                x_n = np.reshape(X[i][:, n], (self.d[i], 1))
                z_n = np.reshape(self.means_z[:, n], (self.m, 1))
                S += np.dot(x_n, x_n.T)
                S -= np.dot(x_n, np.dot(z_n.T, self.means_w[i].T))
                S -= np.dot(x_n, self.means_mu[i].T)
                S -= np.dot(self.means_w[i], z_n).dot(x_n.T)
                S += np.dot(self.means_w[i], z_n).dot(self.means_mu[i].T)
                S -= np.dot(self.means_mu[i], x_n.T)
                S += np.dot(self.means_mu[i], np.dot(z_n.T, self.means_w[i].T))
                S += self.sigma_mu[i] + np.dot(self.means_mu[i], self.means_mu[i].T)
            
            self.K_tilde[i] = self.K[i] + S + diag
            self.E_phi[i] = self.nu_tilde[i] * np.linalg.inv(self.K_tilde[i])     

    def lower_bound(self, X):
        ## Compute the lower bound##       
        # ln p(X_n|Z_n,theta)
        L = 0
        logalpha = [[] for _ in range(self.s)]
        logphi = [[] for _ in range(self.s)]
        for i in range(0, self.s):
            # calculate ln alpha
            logalpha[i] = digamma(self.a_ard[i]) - np.log(self.b_ard[i])
            # calculate ln phi 
            PSD_K = np.dot(self.K_tilde[i].T, self.K_tilde[i])/50
            cho_K = np.linalg.cholesky(PSD_K)
            detK = -2 * np.sum(np.log(np.diag(cho_K)))
            logphi[i] = multigammaln(self.nu[i]/2,self.d[i]) + (self.d[i] * np.log(2)) + detK
            L += -0.5 * self.d[i] * np.log(2*np.pi) - logphi[i]                            
            # Traces on the diagonals 
            diag = np.zeros((self.d[i], self.d[i]))
            for j in range(0, self.d[i]):
                diag[j,j] = np.trace(self.E_zz * self.sigma_w[i][:,:,j])        

            S = 0
            for n in range(0, self.N):
                x_n = np.reshape(X[i][:, n], (self.d[i], 1))
                z_n = np.reshape(self.means_z[:, n], (self.m, 1))
                S += np.dot(x_n, x_n.T)
                S -= np.dot(x_n, np.dot(z_n.T, self.means_w[i].T))
                S -= np.dot(x_n, self.means_mu[i].T)
                S -= np.dot(self.means_w[i], z_n).dot(x_n.T)
                S += np.dot(self.means_w[i], z_n).dot(self.means_mu[i].T)
                S -= np.dot(self.means_mu[i], x_n.T)
                S += np.dot(self.means_mu[i], np.dot(z_n.T, self.means_w[i].T))
                S += self.sigma_mu[i] + np.dot(self.means_mu[i], self.means_mu[i].T)
            S += diag            
            L -= 0.5 * np.trace(np.dot(S,self.E_phi[i]))

        # E[ln p(Z)] - E[ln q(Z)]
        self.Lpz = - 1/2 * np.sum(np.diag(self.E_zz))
        self.Lqz = - self.N * 0.5 * (self.detZ + self.m)
        L += self.Lpz - self.Lqz

        # E[ln p(W|alpha)] - E[ln q(W|alpha)]
        for i in range(0, self.s):
            self.Lpw = 0.5 * self.d[i] * np.sum(logalpha[i])
            self.Lqw = - 0.5 * self.d[i] * self.m
            for j in range(0, self.m):
                E_wwj = self.sigma_w[i][:,:,j] + \
                    np.dot(self.means_w[i][:,j], self.means_w[i][:,j].T)     
                self.Lpw -= self.E_alpha[i][:,j] * np.sum(np.diag(E_wwj))
                PSD_W = np.dot(self.sigma_w[i][:,:,j], self.sigma_w[i][:,:,j])/50
                cho_W = np.linalg.cholesky(PSD_W)
                detwwj = -2 * np.sum(np.log(np.diag(cho_W)))
                self.Lqw -= self.m * 0.5 * detwwj 
        L += self.Lpw - self.Lqw                           

        # E[ln p(alpha) - ln q(alpha)]
        self.Lpa = self.Lqa = 0
        for i in range(0, self.s):
            self.Lpa += self.m * (-np.log(gamma(self.a[i])) + self.a[i] * np.log(self.b[i])) \
                - (self.a[i] - 1) * np.sum(logalpha[i]) - self.b[i] * np.sum(self.E_alpha[i])
            self.Lqa -= self.m * (self.a_ard[i] * np.sum(np.log(self.b_ard[i]))) + \
                np.log(gamma(self.a_ard[i])) + ((self.a_ard[i] - 1) * np.sum(logalpha[i])) -\
                    np.sum(self.b_ard[i] * self.E_alpha[i])         
        L += self.Lpa - self.Lqa 

        # E[ln p(mu) - ln q(mu)]
        self.Lpmu = self.Lqmu = 0
        for i in range(0, self.s):
            E_uu = self.d[i] * self.sigma_mu[i] + np.dot(self.means_mu[i], self.means_mu[i].T)
            PSD_mu = np.dot(self.sigma_mu[i].T, self.sigma_mu[i])/50
            cho_mu = np.linalg.cholesky(PSD_mu)
            detmu = -2 * np.sum(np.log(np.diag(cho_mu))) 
            self.Lpmu += 1/2 * np.log(self.beta[i]) - (self.beta[i]/2) * np.sum(np.diag(E_uu)) 
            self.Lqmu -= self.d[i] * 0.5 * (detmu + 1)
        L += self.Lpmu - self.Lqmu               

        # E[ln p(phi) - ln q(phi)]
        self.Lpphi = self.Lqphi = 0
        for i in range(0, self.s):
            PSD_K = np.dot(self.K_tilde[i].T, self.K_tilde[i])/50
            cho_K = np.linalg.cholesky(PSD_K)
            detK = -2 * np.sum(np.log(np.diag(cho_K)))
            self.Lpphi += multigammaln(self.nu[i]/2,self.d[i]) + (self.d[i] * np.log(2)) + detK
            self.Lqphi += 0.5 * (self.d[i] + 1) * detK + self.d[i]/2 * (self.d[i] + 1) * np.log(2) \
                + multigammaln(self.nu[i]/2,self.d[i]) - 1/2 * (self.nu_tilde[i] - self.d[i] - 1) \
                    * digamma(multigammaln(self.nu[i]/2, self.d[i])) + (self.nu_tilde[i] * self.d[i])/2
        L += self.Lpphi - self.Lqphi

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
            print("Lower Bound Value : %d", self.lower_bound(X))
            L_new = self.lower_bound(X)
            L_mat.append(L_new)
            if abs(L_new - L_previous) < threshold:
                break
            L_previous = L_new
        return L_mat

    def scale_rows(self, X,s):
        y = np.dot(repmat(s,1,X.shape[1]),X.T)

        return y