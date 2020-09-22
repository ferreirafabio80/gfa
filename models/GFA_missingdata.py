"""Group Factor Analysis"""

#Author: Fabio S. Ferreira (fabio.ferreira.16@ucl.ac.uk)
#Date: 17 September 2020

import numpy as np
from numpy.matlib import repmat
from scipy.special import digamma, gammaln
from scipy.optimize import fmin_l_bfgs_b as lbfgsb

class GFA_DiagonalNoiseModel(object):

    def __init__(self, X, args, imputation=False):
        self.s = args.num_sources #number of data sources/groups
        self.d = np.array([X[0].shape[1], X[1].shape[1]])  #number of features in each group
        self.td = np.sum(self.d) #total number of features
        self.k = args.K   #number of components
        self.N = X[0].shape[0] #number of samples
        # Check scenario ('complete' for complete data; 'incomplete' for incomplete data)
        if imputation:
            self.scenario = 'complete'
        else:
            self.scenario = args.scenario

        #hyperparameters
        self.a0_alpha = self.b0_alpha = self.a0_tau = self.b0_tau = np.array([1e-14, 1e-14])

        # Initialising variational parameters
        #Latent variables
        self.means_z = np.reshape(np.random.normal(0, 1, self.N*self.k),(self.N, self.k))
        self.sigma_z = np.zeros((self.k, self.k, self.N))
        self.sum_sigmaZ = self.N * np.identity(self.k)
        #Loading matrices
        self.means_w = [[] for _ in range(self.s)]
        self.sigma_w = [[] for _ in range(self.s)]
        self.E_WW = [[] for _ in range(self.s)]
        self.Lqw = [[] for _ in range(self.s)]
        #Alpha parameters
        self.a_alpha = [[] for _ in range(self.s)]
        self.b_alpha = [[] for _ in range(self.s)]
        self.E_alpha = [[] for _ in range(self.s)]
        #Noise parameters
        self.a_tau = [[] for _ in range(self.s)]
        self.b_tau = [[] for _ in range(self.s)]
        self.E_tau = [[] for _ in range(self.s)]
        if self.scenario == 'incomplete':
            #initialise variables to incomplete data sets
            self.X_nan = [[] for _ in range(self.s)]
            self.N_clean = [[] for _ in range(self.s)]
        #Constants
        self.logalpha = [[] for _ in range(self.s)]
        self.logtau = [[] for _ in range(self.s)]
        self.L_const = [[] for _ in range(self.s)]
        for i in range(0, self.s):
            if self.scenario == 'incomplete':
                # Checking NaNs
                X_new = np.zeros((1, X[i].size))
                X_new[0, np.flatnonzero(np.isnan(X[i]))] = 1
                self.X_nan[i] = np.reshape(X_new,(self.N, self.d[i]))
                self.N_clean[i] = np.sum(~np.isnan(X[i]),axis=0)
            #loading matrices
            self.means_w[i] = np.zeros((self.d[i], self.k))
            self.sigma_w[i] = np.zeros((self.k,self.k,self.d[i])) 
            #alpha parameters
            self.a_alpha[i] = self.a0_alpha[i] + self.d[i]/2.0
            self.b_alpha[i] = np.ones((1, self.k))
            self.E_alpha[i] = self.a_alpha[i] / self.b_alpha[i] 
            #noise parameters
            if self.scenario == 'incomplete':
                self.a_tau[i] = self.a0_tau[i] + (self.N_clean[i])/2
            else:
                self.a_tau[i] = self.a0_tau[i] + (self.N) * np.ones((1,self.d[i]))/2    
            self.b_tau[i] = np.zeros((1, self.d[i]))
            self.E_tau[i] = 1000.0 * np.ones((1, self.d[i]))
            #ELBO constant
            self.L_const[i] = -0.5 * np.sum(self.N_clean[i]) * np.log(2*np.pi)
        # Rotation parameters
        self.DoRotation = True

    def update_w(self, X):
        self.sum_sigmaW = [np.zeros((self.k,self.k)) for _ in range(self.s)]
        for i in range(0, self.s):
            self.Lqw[i] = np.zeros((1, self.d[i]))
            if self.scenario == 'complete':
                S1 = self.sum_sigmaZ + np.dot(self.means_z.T,self.means_z) 
                S2 = np.dot(X[i].T,self.means_z)
                for j in range(0, self.d[i]):
                    # Update covariance matrices of Ws    
                    self.sigma_w[i][:,:,j] = np.diag(self.E_alpha[i]) + \
                        self.E_tau[i][0,j] * S1
                    cho = np.linalg.cholesky(self.sigma_w[i][:,:,j])
                    invCho = np.linalg.inv(cho)
                    self.sigma_w[i][:,:,j] = np.dot(invCho.T,invCho)
                    self.sum_sigmaW[i] += self.sigma_w[i][:,:,j]
                    
                    # Update expectations of Ws
                    self.means_w[i][j,:] = np.dot(S2[j,:],self.sigma_w[i][:,:,j]) * \
                        self.E_tau[i][0,j]
                    
                    # Compute determinant for ELBO    
                    self.Lqw[i][0,j] = -2 * np.sum(np.log(np.diag(cho)))
            else:    
                for j in range(0, self.d[i]):
                    samples = np.array(np.where(self.X_nan[i][:,j] == 0))
                    x = np.reshape(X[i][samples[0,:], j],(1, samples.shape[1]))
                    Z = np.reshape(self.means_z[samples[0,:],:],(samples.shape[1],self.k))
                    S1 = self.sum_sigmaZ + np.dot(Z.T,Z) 
                    S2 = np.dot(x,Z)   
                    
                    # Update covariance matrices of Ws    
                    self.sigma_w[i][:,:,j] = np.diag(self.E_alpha[i]) + \
                        self.E_tau[i][0,j] * S1
                    #efficient way of computing sigmaW_j    
                    cho = np.linalg.cholesky(self.sigma_w[i][:,:,j])
                    invCho = np.linalg.inv(cho)
                    self.sigma_w[i][:,:,j] = np.dot(invCho.T,invCho)
                    self.sum_sigmaW[i] += self.sigma_w[i][:,:,j]
                    
                    # Update expectations of Ws
                    self.means_w[i][j,:] = np.dot(S2,self.sigma_w[i][:,:,j]) * \
                        self.E_tau[i][0,j]
                    
                    # Compute determinant for ELBO
                    self.Lqw[i][0,j] = -2 * np.sum(np.log(np.diag(cho)))
            # Calculate E[W^T W]
            self.E_WW[i] = self.sum_sigmaW[i] + \
                    np.dot(self.means_w[i].T, self.means_w[i])

    def update_z(self, X):
        self.means_z = self.means_z * 0
        if self.scenario == 'complete':
            
            # Update covariance matrix of Z
            self.sigma_z = np.identity(self.k)
            for i in range(0, self.s):
                for j in range(0, self.d[i]):
                    w = np.reshape(self.means_w[i][j,:], (1,self.k))
                    ww = self.sigma_w[i][:,:, j] + np.dot(w.T, w)
                    self.sigma_z += ww * self.E_tau[i][0,j]
            #efficient way of computing sigmaZ      
            cho = np.linalg.cholesky(self.sigma_z)
            invCho = np.linalg.inv(cho)
            self.sigma_z = np.dot(invCho.T,invCho)
            self.sum_sigmaZ = self.N * self.sigma_z
            
            # Compute determinant for ELBO  
            self.Lqz = -2 * np.sum(np.log(np.diag(cho)))  
            
            # Update expectations of Z
            self.means_z = self.means_z * 0
            for i in range(0, self.s):
                for j in range(0, self.d[i]):
                    x = np.reshape(X[i][:, j],(self.N,1))
                    w = np.reshape(self.means_w[i][j,:], (1,self.k))
                    self.means_z += np.dot(x, w) * self.E_tau[i][0,j]
            self.means_z = np.dot(self.means_z, self.sigma_z)
        else:
            self.sigma_z = np.zeros((self.k,self.k,self.N))
            self.sum_sigmaZ = np.zeros((self.k,self.k))
            self.Lqz = np.zeros((1, self.N))
            for n in range(0, self.N):
                self.sigma_z[:,:,n] = np.identity(self.k)
                S1 = np.zeros((1,self.k))  
                for i in range(0, self.s):             
                    dim = np.array(np.where(self.X_nan[i][n,:] == 0))                           
                    for j in range(dim.shape[1]):
                        w = np.reshape(self.means_w[i][dim[0,j],:], (1,self.k))
                        ww = self.sigma_w[i][:,:, dim[0,j]] + np.dot(w.T, w)
                        self.sigma_z[:,:,n] += ww * self.E_tau[i][0,dim[0,j]]
                    x = np.reshape(X[i][n, dim[0,:]],(1, dim.size))
                    tau = np.reshape(self.E_tau[i][0,dim[0,:]],(1, dim.size))
                    S1 += np.dot(x, np.diag(tau[0])).dot(self.means_w[i][dim[0,:],:])
                
                # Update covariance matrix of Z    
                cho = np.linalg.cholesky(self.sigma_z[:,:,n])
                invCho = np.linalg.inv(cho)
                self.sigma_z[:,:,n] = np.dot(invCho.T,invCho)
                self.sum_sigmaZ += self.sigma_z[:,:,n]
                
                # Update expectations of Z
                self.means_z[n,:] = np.dot(S1, self.sigma_z[:,:,n])
                
                # Compute determinant for ELBO
                self.Lqz[0,n] = -2 * np.sum(np.log(np.diag(cho)))     
        self.E_zz = self.sum_sigmaZ + np.dot(self.means_z.T, self.means_z)     

    def update_alpha(self):
        for i in range(0, self.s):
            ## Update b_alpha
            self.b_alpha[i] = self.b0_alpha[i] + np.diag(self.E_WW[i])/2
            ## Update expectation of alpha
            self.E_alpha[i] = self.a_alpha[i] / self.b_alpha[i]         

    def update_tau(self, X):
        ## Update parameters for tau
        for i in range(0, self.s):   
            for j in range(0, self.d[i]):
                if self.scenario == 'complete':
                    w = np.reshape(self.means_w[i][j,:], (1,self.k))
                    ww = self.sigma_w[i][:,:, j] + np.dot(w.T, w)
                    x = np.reshape(X[i][:, j],(self.N,1))
                    z = self.means_z
                    ZZ = self.E_zz
                else:
                    samples = np.array(np.where(self.X_nan[i][:,j] == 0))
                    w = np.reshape(self.means_w[i][j,:], (1,self.k))
                    ww = self.sigma_w[i][:,:,j] + np.dot(w.T,w)
                    x = np.reshape(X[i][samples[0,:],j],(samples.size,1)) 
                    z = np.reshape(self.means_z[samples[0,:],:],(samples.size,self.k)) 
                    sum_covZ = np.sum(self.sigma_z[:,:,samples[0,:]],axis=2) 
                    ZZ = sum_covZ + np.dot(z.T,z) 
                ## Update b_tau        
                self.b_tau[i][0,j] = self.b0_tau[i] + 0.5 * (np.dot(x.T,x) + \
                    np.trace(np.dot(ww, ZZ)) - 2 * np.dot(x.T,z).dot(w.T)) 
            ## Update expectation of tau             
            self.E_tau[i] = self.a_tau[i]/self.b_tau[i]           

    def lower_bound(self, X):
        ## Compute the lower bound##       
        # ln p(X_n|Z_n,theta)
        L = 0
        for i in range(0, self.s):
            # calculate ln alpha
            self.logalpha[i] = digamma(self.a_alpha[i]) - np.log(self.b_alpha[i])
            self.logtau[i] = digamma(self.a_tau[i]) - np.log(self.b_tau[i])
            if self.scenario == 'complete':
                L += self.L_const[i] + np.sum(self.N * self.logtau[i]) / 2 - \
                np.sum(self.E_tau[i] * (self.b_tau[i] - self.b0_tau[i])) 
            else:    
                L += self.L_const[i] + np.sum(self.N_clean[i] * self.logtau[i]) / 2 - \
                    np.sum(self.E_tau[i] * (self.b_tau[i] - self.b0_tau[i]))   

        # E[ln p(Z)] - E[ln q(Z)]
        self.Lpz = - 1/2 * np.sum(np.diag(self.E_zz))
        if self.scenario == 'complete':
            self.Lqz = - self.N * 0.5 * (self.Lqz + self.k)
        else: 
            self.Lqz = - 0.5 * (np.sum(self.Lqz) + self.k)   
        L += self.Lpz - self.Lqz

        # E[ln p(W|alpha)] - E[ln q(W|alpha)]
        self.Lpw = 0
        for i in range(0, self.s):
            self.Lpw += 0.5 * self.d[i] * np.sum(self.logalpha[i]) - np.sum(
                np.diag(self.E_WW[i]) * self.E_alpha[i])
            self.Lqw[i] = - 0.5 * np.sum(self.Lqw[i]) - 0.5 * self.d[i] * self.k 
        L += self.Lpw - sum(self.Lqw)                           

        # E[ln p(alpha) - ln q(alpha)]
        self.Lpa = self.Lqa = 0
        for i in range(0, self.s):
            self.Lpa += self.k * (-gammaln(self.a0_alpha[i]) + self.a0_alpha[i] * np.log(self.b0_alpha[i])) \
                + (self.a0_alpha[i] - 1) * np.sum(self.logalpha[i]) - self.b0_alpha[i] * np.sum(self.E_alpha[i])
            self.Lqa += -self.k * gammaln(self.a_alpha[i]) + self.a_alpha[i] * np.sum(np.log(
                self.b_alpha[i])) + ((self.a_alpha[i] - 1) * np.sum(self.logalpha[i])) - \
                np.sum(self.b_alpha[i] * self.E_alpha[i])         
        L += self.Lpa - self.Lqa               

        # E[ln p(tau) - ln q(tau)]
        self.Lpt = self.Lqt = 0
        for i in range(0, self.s):
            self.Lpt +=  self.d[i] * (-gammaln(self.a0_tau[i]) + self.a0_tau[i] * np.log(self.b0_tau[i])) \
                + (self.a0_tau[i] -1) * np.sum(self.logtau[i]) - self.b0_tau[i] * np.sum(self.E_tau[i])
            self.Lqt += -np.sum(gammaln(self.a_tau[i])) + np.sum(self.a_tau[i] * np.log(self.b_tau[i])) + \
                np.sum((self.a_tau[i] - 1) * self.logtau[i]) - np.sum(self.b_tau[i] * self.E_tau[i])         
        L += self.Lpt - self.Lqt

        return L

    def fit(self, X, iterations=10000, threshold=1e-6):
        L_previous = 0
        self.L = []
        for i in range(iterations):           
            self.remove_components()
            self.update_w(X)
            self.update_z(X)
            if i > 0 and self.DoRotation == True:
                self.update_Rot()   
            self.update_alpha()
            self.update_tau(X)                
            L_new = self.lower_bound(X)
            self.L.append(L_new)
            diff = L_new - L_previous
            if abs(diff)/abs(L_new) < threshold:
                print("ELBO (last value):", L_new)
                print("Number of iterations:", i+1)
                self.iter = i+1
                break
            elif i == iterations:
                print("ELBO did not converge")
            L_previous = L_new
            if i < 1:
                print("ELBO (1st value):", L_new)

    def update_Rot(self):
        ## Update Rotation 
        r = np.matrix.flatten(np.identity(self.k))
        r_opt = lbfgsb(self.Er, r, self.gradEr)
    
        if r_opt[2]['warnflag'] == 0:
            Rot = np.reshape(r_opt[0],(self.k,self.k))
            u, s, v = np.linalg.svd(Rot) 
            Rotinv = np.dot(v.T * np.outer(np.ones((1,self.k)), 1/s), u.T)
            det = np.sum(np.log(s)) 
            
            self.means_z = np.dot(self.means_z, Rotinv.T)
            if self.scenario == 'complete':
                self.sigma_z = np.dot(Rotinv, self.sigma_z).dot(Rotinv.T) 
            else:    
                for n in range(0, self.N):
                    self.sigma_z[:,:,n] = np.dot(Rotinv, self.sigma_z[:,:,n]).dot(Rotinv.T)
                    self.sum_sigmaZ += self.sigma_z[:,:,n]
            self.E_zz = self.sum_sigmaZ + np.dot(self.means_z.T, self.means_z) 
            self.Lqz += -2 * det  

            self.sum_sigmaW = [np.zeros((self.k,self.k)) for _ in range(self.s)]
            for i in range(0, self.s):
                self.means_w[i] = np.dot(self.means_w[i], Rot)
                for j in range(0, self.d[i]):
                    self.sigma_w[i][:,:,j] = np.dot(Rot.T, 
                        self.sigma_w[i][:,:,j]).dot(Rot)
                    self.sum_sigmaW[i] += self.sigma_w[i][:,:,j]     
                self.E_WW[i] = self.sum_sigmaW[i] + \
                    np.dot(self.means_w[i].T, self.means_w[i])
                self.Lqw[i] += 2 * det
        else:
            self.DoRotation = False    
            print('Rotation stopped')     

    def Er(self, r):
        
        R = np.reshape(r,(self.k,self.k))
        u, s, v = np.linalg.svd(R)
        tmp = u * np.outer(np.ones((1,self.k)), 1/s)
        val = -0.5 * np.sum(self.E_zz * np.dot(tmp,tmp.T))
        val += (self.td - self.N) * np.sum(np.log(s))
        for i in range(0, self.s):
            tmp = R * np.dot(self.E_WW[i],R)
            val -= self.d[i] * np.sum(np.log(np.sum(tmp,axis=0)))/2
        val = - val   
        return val

    def gradEr(self, r):
        R = np.reshape(r,(self.k,self.k))
        u, s, v = np.linalg.svd(R) 
        Rinv = np.dot(v.T * np.outer(np.ones((1,self.k)), 1/s), u.T)
        tmp = u * np.outer(np.ones((1,self.k)), 1/(s ** 2)) 
        tmp1 = np.dot(tmp, u.T).dot(self.E_zz) + \
            np.diag((self.td - self.N) * np.ones((1,self.k))[0])
        grad = np.matrix.flatten(np.dot(tmp1, Rinv.T))
        
        for i in range(0, self.s):
            A = np.dot(self.E_WW[i],R)
            B = 1/np.sum((R * A),axis=0)
            tmp2 = self.d[i] * np.matrix.flatten(A * \
                np.outer(np.ones((1,self.k)), B))
            grad -= tmp2
        grad = - grad
        return grad        
    
    def remove_components(self):
        colMeans_Z = np.mean(self.means_z ** 2, axis=0)
        cols_rm = np.ones(colMeans_Z.shape[0], dtype=bool)
    
        if any(colMeans_Z < 1e-6):
            cols_rm[colMeans_Z < 1e-6] = False
            self.means_z = self.means_z[:,cols_rm]
            self.sum_sigmaZ = self.sum_sigmaZ[:,cols_rm]
            self.sum_sigmaZ = self.sum_sigmaZ[cols_rm,:]
            self.k = self.means_z.shape[1]

            for i in range(0, self.s):
                self.means_w[i] = self.means_w[i][:,cols_rm]
                self.sigma_w[i] = self.sigma_w[i][:,cols_rm,:]
                self.sigma_w[i] = self.sigma_w[i][cols_rm,:,:]
                self.E_WW[i] = self.E_WW[i][:,cols_rm]
                self.E_WW[i] = self.E_WW[i][cols_rm,:]
                self.E_alpha[i] = self.E_alpha[i][cols_rm]   
