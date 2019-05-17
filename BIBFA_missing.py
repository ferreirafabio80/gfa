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

        ## Hyperparameters
        self.a = self.b = self.a0_tau = self.b0_tau = np.array([1e-14, 1e-14])
        self.beta = np.array([1e-03, 1e-03])
        self.E_tau = np.array([1e03, 1e03])

        ## Initialising variational parameters
        # Latent variables
        self.sigma_z = np.identity(m)
        #self.means_z  = np.ones((self.N, m))
        self.means_z = np.reshape(np.random.normal(0, 1, self.N*m),(self.N, m))
        # Projection matrices
        self.means_w = [[] for _ in range(self.s)]
        self.sigma_w = [[] for _ in range(self.s)]
        self.E_WW = [[] for _ in range(self.s)]
        self.Lqw = [0 for _ in range(self.s)]
        # ARD parameters (Gamma distribution)
        #-the parameters for the ARD precisions
        self.a_ard = [[] for _ in range(self.s)]
        self.b_ard = [[] for _ in range(self.s)]
        #-the mean of the ARD precisions
        self.E_alpha = [[] for _ in range(self.s)]
        # Precisions (Gamma distribution)
        self.a_tau = [[] for _ in range(self.s)]
        self.b_tau = [[] for _ in range(self.s)]
        for i in range(0, self.s):
            self.means_w[i] = np.reshape(np.random.normal(0, 1, d[i]*self.m),(d[i], self.m))
            self.sigma_w[i] = np.identity(m)
            self.a_ard[i] = self.a[i] + d[i]/2.0
            self.b_ard[i] = np.ones((1, self.m))
            self.a_tau[i] = self.a0_tau[i] + (self.N * self.d[i])/2
            self.b_tau[i] = np.zeros((1, d[i]))
            self.E_alpha[i] = self.a_ard[i] / self.b_ard[i]  

        # Rotation parameters
        self.Rot = np.identity(m)
        self.RotInv = np.identity(m)
        self.r = np.matrix.flatten(self.Rot)

    def update_w(self, X):
        for i in range(0, self.s):     
            ## Update covariance matrices of Ws
            S1 = np.zeros((self.m, self.m))
            S2 = np.zeros((self.d[i], self.m))
            X_new = np.zeros((1, X[i].size))
            X_new[0, np.flatnonzero(np.isnan(X[i]))] = 1
            X_mat = np.reshape(X_new,(self.N, self.d[i]))
            check = 0
            for n in range(0, X_mat.shape[0]):
                z_n = np.reshape(self.means_z[n,:],(1,self.m))
                for m in range(0, X_mat.shape[1]): 
                    if X_mat[n,m] == 0:
                        S2[m,:] += (z_n * X[i][n, m])[0]
                        check = 1
                if check == 1:
                    S1 += self.sigma_z + np.dot(z_n.T,z_n)
                    check = 0        

            ## Update covariance matrix of Z              
            tmp = 1/np.sqrt(self.E_alpha[i]) 
            cho = np.linalg.cholesky((np.outer(tmp, tmp) * S1) + 
                (np.identity(self.m) * (1/self.E_tau[i])))
            # Determinant for lower bound    
            detW = -2 * np.sum(np.log(np.diag(cho))) - np.sum(
                np.log(self.E_alpha[i])) - (self.m * np.log(self.E_tau[i]))
            self.Lqw[i] = detW
            invCho = np.linalg.inv(cho)
            self.sigma_w[i] = 1/self.E_tau[i] * np.outer(tmp, tmp) * \
                np.dot(invCho.T,invCho) 

            np.dot(X[i].T,self.means_z)
            ## Update expectations of Ws
            self.means_w[i] = np.dot(S2, self.sigma_w[i]) * self.E_tau[i]    
            
            self.E_WW[i] = self.d[i] * self.sigma_w[i] + \
                np.dot(self.means_w[i].T, self.means_w[i])

    def update_z(self, X):
        self.sigma_z = np.identity(self.m)
        self.means_z = self.means_z * 0
        S2 = [np.zeros((self.N, self.m)) for _ in range(self.s)]    
        for i in range(0, self.s):  
            S1 = np.zeros((self.m, self.m))            
            X_new = np.zeros((1, X[i].size))
            X_new[0, np.flatnonzero(np.isnan(X[i]))] = 1
            X_mat = np.reshape(X_new,(self.N, self.d[i]))
            check = 0
            for m in range(0, X_mat.shape[1]):
                w_m = np.reshape(self.means_w[i][m,:], (1,self.m))        
                for n in range(0, X_mat.shape[0]): 
                    if X_mat[n,m] == 0:  
                        S2[i][n,:] += (w_m * X[i][n, m])[0]
                        check = 1
                if check == 1:
                    S1 += self.sigma_w[i] + np.dot(w_m.T,w_m)
                    check = 0 
            
            ## Update covariance matrix of Z   
            self.sigma_z += self.E_tau[i] * S1

        cho = np.linalg.cholesky(self.sigma_z)
        self.Lqz = self.detZ = -2 * np.sum(np.log(np.diag(cho)))
        invCho = np.linalg.inv(cho)
        self.sigma_z = np.dot(invCho.T,invCho)

        np.dot(X[i], self.means_w[i])    
        ## Update expectations of Z                      
        for i in range(0, self.s): 
            self.means_z += self.E_tau[i] * np.dot(S2[i], self.sigma_z)
        self.E_zz = self.N * self.sigma_z + np.dot(self.means_z.T, self.means_z)     

    def update_alpha(self):
        for i in range(0, self.s):
            ## Update b
            self.b_ard[i] = self.b[i] + np.diag(self.E_WW[i])/2
            self.E_alpha[i] = self.a_ard[i] / self.b_ard[i]         

    def update_tau(self, X):
        for i in range(0, self.s):         
            ## Update tau
            X_new = np.zeros((1, X[i].size))
            X_new[0, np.flatnonzero(np.isnan(X[i]))] = 1
            X_mat = np.reshape(X_new,(self.N, self.d[i]))
            S = 0
            S1 = np.zeros((self.N, self.m))
            for m in range(0, X_mat.shape[1]):
                w_m = np.reshape(self.means_w[i][m,:], (1,self.m)) 
                for n in range(0, X_mat.shape[0]): 
                    if X_mat[n,m] == 0:
                        S += X[i][n, m] ** 2
                        z_n = self.means_z[n,:]
                        S1[n,:] += (w_m * X[i][n, m])[0] * z_n

            self.b_tau[i] = self.b0_tau[i] + 0.5 * (S + \
                np.sum(self.E_WW[i] * self.E_zz) - 2 * np.sum(S1))
            self.E_tau[i] = self.a_tau[i]/self.b_tau[i]

    def update_Rot(self):
        ## Update Rotation 
        r = np.matrix.flatten(np.identity(self.m))
        r_opt = lbfgsb(self.Er, r, self.gradEr)
        
        Rot = np.reshape(r_opt[0],(self.m,self.m))
        u, s, v = np.linalg.svd(Rot) 
        Rotinv = np.dot(v * np.outer(np.ones((1,self.m)), 1/s), u.T)
        det = np.sum(np.log(s))
        self.means_z = np.dot(self.means_z, Rotinv.T)
        self.sigma_z = np.dot(Rotinv, self.sigma_z).dot(Rotinv.T)
        self.E_zz = self.N * self.sigma_z + np.dot(self.means_z.T, self.means_z) 
        self.Lqz += -2 * det  

        for i in range(0, self.s):
            self.means_w[i] = np.dot(self.means_w[i], Rot)
            self.sigma_w[i] = np.dot(Rot.T, self.sigma_w[i]).dot(Rot)
            self.E_WW[i] = self.d[i] * self.sigma_w[i] + \
                np.dot(self.means_w[i].T, self.means_w[i])
            self.Lqw[i] += 2 * det        

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

    def fit(self, X, iterations=10000, threshold=1e-5):
        L_previous = 0
        L = []
        for i in range(iterations):
            self.update_w(X)            
            self.update_z(X)
            #if i > 0:
                #self.update_Rot() 
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
            #print("Lower Bound Value:", L_new)
        return L

    def Er(self, r):
        
        R = np.reshape(r,(self.m,self.m))
        u, s, v = np.linalg.svd(R)
        tmp = u * np.outer(np.ones((1,self.m)), 1/s)
        val = -0.5 * np.sum(self.E_zz * np.dot(tmp,tmp.T))
        val += (self.td - self.N) * np.sum(np.log(s))
        for i in range(0, self.s):
            tmp = R * np.dot(self.E_WW[i],R)
            val += -self.d[i] * np.sum(np.log(np.sum(tmp,0)))/2
        val = - val   
        return val

    def gradEr(self, r):
        R = np.reshape(r,(self.m,self.m))
        u, s, v = np.linalg.svd(R) 
        Rinv = np.dot(v * np.outer(np.ones((1,self.m)), 1/s), u.T)
        tmp = u * np.outer(np.ones((1,self.m)), 1/(s ** 2)) 
        tmp1 = np.dot(tmp, u.T).dot(self.E_zz) + np.diag((self.td - self.N) * np.ones((1,self.m))[0])
        grad = np.matrix.flatten(np.dot(tmp1, Rinv.T))
        
        for i in range(0, self.s):
            A = np.dot(self.E_WW[i],R)
            B = 1/np.sum(R*A,0)
            tmp2 = self.d[i] * np.matrix.flatten(A * np.outer(np.ones((1,self.m)), B))
            grad += -tmp2
        grad = - grad
        return grad        

