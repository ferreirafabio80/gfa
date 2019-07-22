import numpy as np
import matplotlib.pyplot as plt
from numpy.matlib import repmat
from scipy.special import digamma
from scipy.special import gammaln
from scipy.optimize import fmin_l_bfgs_b as lbfgsb

class BIBFA(object):

    def __init__(self, X, m, d):

        self.s = d.size # number of sources
        self.d = d  # dimensions of data sources
        self.td = np.sum(d) #total number of features
        self.m = m   # number of different models
        self.N = X[0].shape[0]  # data points
        self.N_clean = np.sum(~np.isnan(X[0]),axis=0)
        self.E_tau = np.array([1e03, 1e03]) 

        ## Hyperparameters
        self.a = self.b = self.a0_tau = self.b0_tau = np.array([1e-14, 1e-14])
        self.beta = np.array([1e-03, 1e-03])

        ## Initialising variational parameters
        # Latent variables
        self.means_z = np.reshape(np.random.normal(0, 1, self.N*m),(self.N, m))
        self.sigma_z = np.zeros((m,m,self.N))
        for n in range(0, self.N):
            self.sigma_z[:,:,n] = np.identity(m)
        # Projection matrices
        self.means_w = [[] for _ in range(self.s)]
        self.sigma_w = [[] for _ in range(self.s)]
        self.E_WW = [[] for _ in range(self.s)]
        self.Lqw = [[] for _ in range(self.s)]
        # ARD parameters (Gamma distribution)
        #-the parameters for the ARD parameters
        self.a_ard = [[] for _ in range(self.s)]
        self.b_ard = [[] for _ in range(self.s)]
        #-the mean of the ARD parameters
        self.E_alpha = [[] for _ in range(self.s)]
        # Precisions (Gamma distribution)
        self.a_tau = [[] for _ in range(self.s)]
        self.b_tau = [[] for _ in range(self.s)]
        # NaNs
        self.X_nan = [[] for _ in range(self.s)]
        for i in range(0, self.s):
            #projections
            self.means_w[i] = np.reshape(np.random.normal(0, 1, d[i]*self.m),(d[i], self.m))
            self.sigma_w[i] = np.zeros((m,m,d[i]))
            #ARD parameters
            self.a_ard[i] = self.a[i] + d[i]/2.0
            self.b_ard[i] = np.ones((1, self.m))
            self.E_alpha[i] = self.a_ard[i] / self.b_ard[i] 
            #noise variances
            self.b_tau[i] = np.ones((1, d[i]))
            self.a_tau[i] = self.a0_tau[i] + (self.N * self.d[i])/2
            #self.E_tau[i] = self.a_tau[i] / self.b_tau[i]
            # Checking NaNs
            X_new = np.zeros((1, X[i].size))
            X_new[0, np.flatnonzero(np.isnan(X[i]))] = 1
            self.X_nan[i] = np.reshape(X_new,(self.N, self.d[i]))   

        # Rotation parameters
        self.Rot = np.identity(m)
        self.RotInv = np.identity(m)
        self.r = np.matrix.flatten(self.Rot)

    def update_w(self, X):
        self.sum_sigmaW = [np.zeros((self.m,self.m)) for _ in range(self.s)]
        for i in range(0, self.s):      
            for j in range(0, self.X_nan[i].shape[1]):
                S1 = np.zeros((self.m, self.m))
                S2 = 0 
                for n in range(0, self.X_nan[i].shape[0]):
                    z_n = np.reshape(self.means_z[n,:],(1,self.m))
                    if self.X_nan[i][n,j] == 0:
                        S2 += (z_n * X[i][n, j])[0]
                        S1 += self.sigma_z[:,:,n] + np.dot(z_n.T,z_n)        
                
                ## Update covariance matrices of Ws
                self.sigma_w[i][:,:,j] = np.diag(self.E_alpha[i]) + self.E_tau[i] * S1
                cho = np.linalg.cholesky(self.sigma_w[i][:,:,j])
                invCho = np.linalg.inv(cho)
                self.sigma_w[i][:,:,j] = np.dot(invCho.T,invCho)
                self.sum_sigmaW[i] += self.sigma_w[i][:,:,j] 
                ## Update expectations of Ws
                self.means_w[i][j,:] = np.dot(S2.T,self.sigma_w[i][:,:,j]) * \
                    self.E_tau[i]

            self.Lqw[i] = -2 * np.sum(np.log(np.diag(cho)))
            self.E_WW[i] = self.sum_sigmaW[i] + \
                    np.dot(self.means_w[i].T, self.means_w[i])

    def update_z(self, X):
        self.sigma_z = np.zeros((self.m,self.m,self.N))
        for n in range(0, self.N):
            self.sigma_z[:,:,n] = np.identity(self.m)
        self.means_z = self.means_z * 0
        S = [np.zeros((self.N,self.m)) for _ in range(self.s)]
        S1 = [np.zeros((self.m,self.m,self.N)) for _ in range(self.s)]
        for i in range(0, self.s):         
            for n in range(0, self.X_nan[i].shape[0]):    
                for j in range(0, self.X_nan[i].shape[1]):
                    w = np.reshape(self.means_w[i][j,:], (1,self.m))
                    ww = self.sigma_w[i][:,:,j] + np.dot(w.T, w) 
                    if self.X_nan[i][n,j] == 0:  
                        S[i][n,:] += self.means_w[i][j,:] * X[i][n, j] 
                        S1[i][:,:,n] += ww 

        self.sum_sigmaZ = np.zeros((self.m,self.m))        
        for n in range(0, self.N):
            for i in range(0, self.s):
                self.sigma_z[:,:,n] += S1[i][:,:,n] * self.E_tau[i]
                self.means_z[n,:] += self.E_tau[i] * S[i][n,:]        
            cho = np.linalg.cholesky(self.sigma_z[:,:,n])
            invCho = np.linalg.inv(cho)
            self.sigma_z[:,:,n] = np.dot(invCho.T,invCho)
            self.sum_sigmaZ += self.sigma_z[:,:,n]
            self.means_z[n,:] = np.dot(self.means_z[n,:], self.sigma_z[:,:,n])

        self.Lqz = -2 * np.sum(np.log(np.diag(cho)))

        ## Update expectations of Z                    
        self.E_zz = self.sum_sigmaZ + np.dot(self.means_z.T, self.means_z)     

    def update_alpha(self):
        for i in range(0, self.s):
            ## Update b
            self.b_ard[i] = self.b[i] + np.diag(self.E_WW[i])/2
            self.E_alpha[i] = self.a_ard[i] / self.b_ard[i]         

    def update_tau(self, X):
        for i in range(0, self.s):   
            ## Update tau
            S=0
            for j in range(0, self.X_nan[i].shape[1]):
                w = np.reshape(self.means_w[i][j,:], (1,self.m))
                sig_w = self.sigma_w[i][:,:,j]
                ww = sig_w + np.dot(w.T,w)
                for n in range(0, self.X_nan[i].shape[0]):
                    if self.X_nan[i][n,j] == 0: 
                        x = X[i][n,j]
                        z = np.reshape(self.means_z[n,:],(1,self.m))
                        sig_z = self.sigma_z[:,:,n] 
                        zz = sig_z + np.dot(z.T,z)
                        S += x ** 2 + np.trace(np.dot(ww, zz)) - \
                            2 * x * np.dot(w,z.T) + np.dot(w,sig_z).dot(w.T) + \
                                np.dot(z,sig_w).dot(z.T) + np.trace(np.dot(sig_z,sig_w))    
            self.b_tau[i] = self.b0_tau[i] + 0.5 * S[0]         
            self.E_tau[i] = self.a_tau[i]/self.b_tau[i]              

    def lower_bound(self, X):
        ## Compute the lower bound##       
        # ln p(X_n|Z_n,theta)
        L = 0
        logalpha = [[] for _ in range(self.s)]
        logtau = [[] for _ in range(self.s)]
        for i in range(0, self.s):
            # calculate ln alpha
            logalpha[i] = digamma(self.a_ard[i]) - np.log(self.b_ard[i])
            logtau[i] = digamma(self.a_tau[i]) - np.log(self.b_tau[i])
            const = -0.5 * self.N * self.d[i] * np.log(2*np.pi)                          
            L += const + self.N * self.d[i] * logtau[i] / 2 - \
                (self.b_tau[i] - self.b0_tau[i]) * self.E_tau[i]   

        # E[ln p(Z)] - E[ln q(Z)]
        self.Lpz = - 1/2 * np.sum(np.diag(self.E_zz))
        self.Lqz = - self.N * 0.5 * (self.Lqz + self.m)
        L += self.Lpz - self.Lqz

        # E[ln p(W|alpha)] - E[ln q(W|alpha)]
        self.Lpw = 0
        for i in range(0, self.s):
            self.Lpw += 0.5 * self.d[i] * np.sum(logalpha[i]) - np.sum(
                np.diag(self.E_WW[i]) * self.E_alpha[i])
            self.Lqw[i] = - self.d[i]/2 * (self.Lqw[i] + self.m) 
        L += self.Lpw - sum(self.Lqw)                           

        # E[ln p(alpha) - ln q(alpha)]
        self.Lpa = self.Lqa = 0
        for i in range(0, self.s):
            self.Lpa += self.m * (-gammaln(self.a[i]) + self.a[i] * np.log(self.b[i])) \
                + (self.a[i] - 1) * np.sum(logalpha[i]) - self.b[i] * np.sum(self.E_alpha[i])
            self.Lqa += -self.m * gammaln(self.a_ard[i]) + self.a_ard[i] * np.sum(np.log(
                self.b_ard[i])) + ((self.a_ard[i] - 1) * np.sum(logalpha[i])) - \
                np.sum(self.b_ard[i] * self.E_alpha[i])         
        L += self.Lpa - self.Lqa               

        # E[ln p(tau) - ln q(tau)]
        self.Lpt = self.Lqt = 0
        for i in range(0, self.s):
            self.Lpt += -gammaln(self.a0_tau[i]) + (self.a0_tau[i] * np.log(self.b0_tau[i])) \
                + ((self.a0_tau[i] - 1) * np.sum(logtau[i])) - (self.b[i] * np.sum(self.E_tau[i]))
            self.Lqt += -gammaln(self.a_tau[i]) + (self.a_tau[i] * np.log(self.b_tau[i])) + \
                ((self.a_tau[i] - 1) * logtau[i]) - (self.b_tau[i] * self.E_tau[i])         
        L += self.Lpt - self.Lqt 

        return L

    def fit(self, X, iterations=10000, threshold=1e-6):
        L_previous = 0
        L = []
        for i in range(iterations):
            self.remove_components()
            self.update_w(X)
            self.update_z(X)
            #self.update_mu(X)
            #if i > 0:
            #    self.update_Rot() 
            self.update_alpha()
            self.update_tau(X)                
            L_new = self.lower_bound(X)
            L.append(L_new)
            diff = L_new - L_previous
            if abs(diff) < threshold:
                print("Iterations:", i+1)
                print("Lower Bound Value:", L_new)
                self.iter = i+1
                break
            elif i == iterations:
                print("Lower bound did not converge")
            L_previous = L_new
            print("Lower Bound Value:", L_new)
        return L

    def update_Rot(self):
        ## Update Rotation 
        r = np.matrix.flatten(np.identity(self.m))
        #r_opt = lbfgsb(self.Er, r, approx_grad=True, factr=1e7)
        r_opt = lbfgsb(self.Er, r, self.gradEr)
        
        Rot = np.reshape(r_opt[0],(self.m,self.m))
        u, s, v = np.linalg.svd(Rot) 
        Rotinv = np.dot(v.T * np.outer(np.ones((1,self.m)), 1/s), u.T)
        det = np.sum(np.log(s))
        
        self.means_z = np.dot(self.means_z, Rotinv.T)
        self.sigma_z = np.dot(Rotinv, self.sigma_z).dot(Rotinv.T)
        self.E_zz = self.N * self.sigma_z + np.dot(self.means_z.T, self.means_z) 
        self.Lqz += -2 * det  

        self.sum_sigmaW = [np.zeros((self.m,self.m)) for _ in range(self.s)]
        for i in range(0, self.s):
            self.means_w[i] = np.dot(self.means_w[i], Rot)
            for j in range(0, self.d[i]):
                self.sigma_w[i][:,:,j] = np.dot(Rot.T, 
                    self.sigma_w[i][:,:,j]).dot(Rot)
                self.sum_sigmaW[i] += self.sigma_w[i][:,:,j]     
            self.E_WW[i] = self.d[i] * self.sum_sigmaW[i] + \
                np.dot(self.means_w[i].T, self.means_w[i])
            self.Lqw[i] += 2 * det 

    def Er(self, r):
        
        R = np.reshape(r,(self.m,self.m))
        u, s, v = np.linalg.svd(R)
        tmp = u * np.outer(np.ones((1,self.m)), 1/s)
        val = -0.5 * np.sum(self.E_zz * np.dot(tmp,tmp.T))
        val += (self.td - self.N) * np.sum(np.log(s))
        for i in range(0, self.s):
            tmp = R * np.dot(self.E_WW[i],R)
            val -= self.d[i] * np.sum(np.log(np.sum(tmp,axis=0)))/2
        val = - val   
        return val

    def gradEr(self, r):
        R = np.reshape(r,(self.m,self.m))
        u, s, v = np.linalg.svd(R) 
        Rinv = np.dot(v.T * np.outer(np.ones((1,self.m)), 1/s), u.T)
        tmp = u * np.outer(np.ones((1,self.m)), 1/(s ** 2)) 
        tmp1 = np.dot(tmp, u.T).dot(self.E_zz) + \
            np.diag((self.td - self.N) * np.ones((1,self.m))[0])
        grad = np.matrix.flatten(np.dot(tmp1, Rinv.T))
        
        for i in range(0, self.s):
            A = np.dot(self.E_WW[i],R)
            B = 1/np.sum((R * A),axis=0)
            tmp2 = self.d[i] * np.matrix.flatten(A * \
                np.outer(np.ones((1,self.m)), B))
            grad -= tmp2
        grad = - grad
        return grad        
    
    def remove_components(self):
        colMeans_Z = np.mean(self.means_z ** 2, axis=0)
        cols_rm = np.ones(colMeans_Z.shape[0], dtype=bool)
    
        if any(colMeans_Z < 1e-7):
            cols_rm[colMeans_Z < 1e-7] = False
            self.means_z = self.means_z[:,cols_rm]
            self.sigma_z = self.sigma_z[:,cols_rm]
            self.sigma_z = self.sigma_z[cols_rm,:]
            self.E_zz = self.E_zz[:,cols_rm]
            self.E_zz = self.E_zz[cols_rm,:]
            self.m = self.means_z.shape[1]

            for i in range(0, self.s):
                self.means_w[i] = self.means_w[i][:,cols_rm]
                self.sigma_w[i] = self.sigma_w[i][:,cols_rm]
                self.sigma_w[i] = self.sigma_w[i][cols_rm,:]
                self.E_WW[i] = self.E_WW[i][:,cols_rm]
                self.E_WW[i] = self.E_WW[i][cols_rm,:]
                self.E_alpha[i] = self.E_alpha[i][cols_rm] 