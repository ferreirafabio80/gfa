""" Group Factor Analysis (original model) """

#Author: Fabio S. Ferreira (fabio.ferreira.16@ucl.ac.uk)
#Date: 17 September 2020

import numpy as np
from numpy.matlib import repmat
from scipy.special import digamma, gammaln
from scipy.optimize import fmin_l_bfgs_b as lbfgsb

class GFA_OriginalModel(object):

    def __init__(self, X, args):
        
        self.s = args.num_sources #number of data sources
        self.d = np.array([X[0].shape[1], X[1].shape[1]]) #number of features in each group 
        self.td = np.sum(self.d) #total number of features
        self.k = args.K #number of components
        self.N = X[0].shape[0]  #number of samples

        #hyperparameters
        self.a0_alpha = self.b0_alpha = self.a0_tau = self.b0_tau = np.array([1e-14, 1e-14])
        self.E_tau = np.array([1000.0, 1000.0])

        # Initialising variational parameters
        #latent variables
        self.means_z = np.reshape(np.random.normal(0, 1, self.N*self.k),(self.N, self.k))
        self.sigma_z = np.identity(self.k)
        self.E_zz = self.N * self.sigma_z + self.sigma_z
        #loading matrices
        self.means_w = [[] for _ in range(self.s)]
        self.sigma_w = [[] for _ in range(self.s)]
        self.E_WW = [[] for _ in range(self.s)]
        self.Lqw = [[] for _ in range(self.s)]
        #ARD parameters
        self.a_alpha = [[] for _ in range(self.s)]
        self.b_alpha = [[] for _ in range(self.s)]
        self.E_alpha = [[] for _ in range(self.s)]
        #noise parameters
        self.a_tau = [[] for _ in range(self.s)]
        self.b_tau = [[] for _ in range(self.s)]
        #Data variance needed for scaling alphas
        self.datavar = [[] for _ in range(self.s)]
        #constants for ELBO
        self.logalpha = [[] for _ in range(self.s)]
        self.logtau = [[] for _ in range(self.s)]
        self.X_squared = [[] for _ in range(self.s)]
        self.L_const = [[] for _ in range(self.s)]
        for i in range(0, self.s):
            #alpha parameters    
            self.a_alpha[i] = self.a0_alpha[i] + self.d[i]/2.0
            self.b_alpha[i] = np.ones((1, self.k))
            #noise parameters
            self.a_tau[i] = self.a0_tau[i] + (self.N * self.d[i])/2
            self.b_tau[i] = np.zeros((1, self.d[i]))
            #Calculate expectation of alpha
            self.datavar[i] = np.sum(X[i].var(0))
            self.E_alpha[i] = repmat(self.k * self.d[i] / 
                (self.datavar[i]-1/self.E_tau[i]), 1, self.k)
            #X squared    
            self.X_squared[i] = np.sum(X[i] ** 2)
            #ELBO constant 
            self.L_const[i] = -0.5 * self.N * self.d[i] * np.log(2*np.pi)    
        #Rotation parameters
        self.DoRotation = True

    def update_w(self, X):
        
        """ 
        Update the variational parameters of the loading matrices.

        Parameters
        ----------
        X : list 
            List of arrays containing the data matrix of each group.          
        
        """
        for i in range(0, self.s):      
            
            # Compute covariance matrix of Ws
            tmp = 1/np.sqrt(self.E_alpha[i])
            cho = np.linalg.cholesky((np.outer(tmp, tmp) * self.E_zz) + 
                (np.identity(self.k) * (1/self.E_tau[i])))
            invCho = np.linalg.inv(cho)
            self.sigma_w[i] = 1/self.E_tau[i] * np.outer(tmp, tmp) * \
               np.dot(invCho.T,invCho)
            # Determinant for ELBO    
            self.Lqw[i] = -2 * np.sum(np.log(np.diag(cho))) - np.sum(
                np.log(self.E_alpha[i])) - (self.k * np.log(self.E_tau[i]))      
                
            # Compute expectations of Ws  
            self.means_w[i]= np.dot(X[i].T,self.means_z).dot(self.sigma_w[i]) * \
                self.E_tau[i]
            # Calculate E[W^T W]    
            self.E_WW[i] = self.d[i] * self.sigma_w[i] + \
                np.dot(self.means_w[i].T, self.means_w[i])

    def update_z(self, X):
        
        """ 
        Update the variational parameters of the latent variables.

        Parameters
        ----------
        X : list 
            List of arrays containing the data matrix of each group.          
        
        """
        # Compute covariance matrix of Z
        self.sigma_z = np.identity(self.k)
        for i in range(0, self.s):
            self.sigma_z += self.E_tau[i] * self.E_WW[i]  
        cho = np.linalg.cholesky(self.sigma_z)
        self.Lqz = -2 * np.sum(np.log(np.diag(cho)))
        invCho = np.linalg.inv(cho)
        self.sigma_z = np.dot(invCho.T,invCho)

        # Compute expectations of Z
        self.means_z = self.means_z * 0
        for i in range(0, self.s):
            self.means_z += np.dot(X[i], self.means_w[i]) * self.E_tau[i]
        self.means_z = np.dot(self.means_z, self.sigma_z)
        # Calculate E[Z^T Z]
        self.E_zz = self.N * self.sigma_z + np.dot(self.means_z.T, self.means_z)     

    def update_alpha(self):
        
        """ 
        Update the variational parameters of the alphas.

        Parameters
        ----------
        X : list 
            List of arrays containing the data matrix of each group.          
        
        """
        for i in range(0, self.s):
            # Compute b and expectations of the alphas
            self.b_alpha[i] = self.b0_alpha[i] + np.diag(self.E_WW[i])/2
            self.E_alpha[i] = self.a_alpha[i] / self.b_alpha[i]         

    def update_tau(self, X):
        
        """ 
        Update the variational parameters of the taus.

        Parameters
        ----------
        X : list 
            List of arrays containing the data matrix of each group.          
        
        """
        for i in range(0, self.s):         
            # Compute b and expectations of the taus
            self.b_tau[i] = self.b0_tau[i] + 0.5 * (self.X_squared[i] + 
                np.sum(self.E_WW[i] * self.E_zz) - 2 * np.sum(np.dot(
                    X[i], self.means_w[i]) * self.means_z)) 
            self.E_tau[i] = self.a_tau[i]/self.b_tau[i]       

    def lower_bound(self, X):
        
        """ 
        Calculate Evidence Lower Bound (ELBO).

        Parameters
        ----------
        X : list 
            List of arrays containing the data matrix of each group.

        Returns
        -------
        L : float
            ELBO.              
        
        """     
        # Calculate E[ln p(X|Z,W,tau)]
        L = 0
        for i in range(0, self.s):
            #calculate E[ln alpha] and E[ln tau]
            self.logalpha[i] = digamma(self.a_alpha[i]) - np.log(self.b_alpha[i])
            self.logtau[i] = digamma(self.a_tau[i]) - np.log(self.b_tau[i])                         
            L += self.L_const[i] + self.N * self.d[i] * self.logtau[i] / 2 - \
                self.d[i] * self.E_tau[i] * (self.b_tau[i] - self.b0_tau[i])    

        # Calculate E[ln p(Z)] - E[ln q(Z)]
        self.Lpz = - 1/2 * np.sum(np.diag(self.E_zz))
        self.Lqz = - self.N * 0.5 * (self.Lqz + self.k)
        L += self.Lpz - self.Lqz

        # Calculate E[ln p(W|alpha)] - E[ln q(W|alpha)]
        self.Lpw = 0
        for i in range(0, self.s):
            self.Lpw += 0.5 * self.d[i] * np.sum(self.logalpha[i]) - np.sum(
                np.diag(self.E_WW[i]) * self.E_alpha[i])
            self.Lqw[i] = - self.d[i]/2 * (self.Lqw[i] + self.k) 
        L += self.Lpw - sum(self.Lqw)                           

        # Calculate E[ln p(alpha) - ln q(alpha)]
        self.Lpa = self.Lqa = 0
        for i in range(0, self.s):
            self.Lpa += self.k * (-gammaln(self.a0_alpha[i]) + self.a0_alpha[i] * np.log(self.b0_alpha[i])) \
                + (self.a0_alpha[i] - 1) * np.sum(self.logalpha[i]) - self.b0_alpha[i] * np.sum(self.E_alpha[i])
            self.Lqa += -self.k * gammaln(self.a_alpha[i]) + self.a_alpha[i] * np.sum(np.log(
                self.b_alpha[i])) + ((self.a_alpha[i] - 1) * np.sum(self.logalpha[i])) - \
                np.sum(self.b_alpha[i] * self.E_alpha[i])         
        L += self.Lpa - self.Lqa               

        # Calculate E[ln p(tau) - ln q(tau)]
        self.Lpt = self.Lqt = 0
        for i in range(0, self.s):
            self.Lpt += -gammaln(self.a0_tau[i]) + (self.a0_tau[i] * np.log(self.b0_tau[i])) \
                + ((self.a0_tau[i] - 1) * np.sum(self.logtau[i])) - (self.b0_tau[i] * np.sum(self.E_tau[i]))
            self.Lqt += -gammaln(self.a_tau[i]) + (self.a_tau[i] * np.log(self.b_tau[i])) + \
                ((self.a_tau[i] - 1) * self.logtau[i]) - (self.b_tau[i] * self.E_tau[i])         
        L += self.Lpt - self.Lqt 

        return L

    def fit(self, X, iterations=10000, thr=1e-6):
        
        """ 
        Fit the original GFA model.

        Parameters
        ----------
        X : list 
            List of arrays containing the data matrix of each group.

        iterations : int
            Maximum number of iterations.

        thr : float
            Threshold to check model convergence. The model stops when 
            a relative difference in the lower bound falls below this 
            value.                     
        
        """
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
            if abs(diff)/abs(L_new) < thr:
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
        
        """ 
        Optimization of the rotation.                    
        
        """
        r = np.matrix.flatten(np.identity(self.k))
        r_opt = lbfgsb(self.Er, r, self.gradEr)
        if r_opt[2]['warnflag'] == 0:
            # Update transformation matrix R
            Rot = np.reshape(r_opt[0],(self.k,self.k))
            u, s, v = np.linalg.svd(Rot) 
            Rotinv = np.dot(v.T * np.outer(np.ones((1,self.k)), 1/s), u.T)
            det = np.sum(np.log(s))

            # Update Z 
            self.means_z = np.dot(self.means_z, Rotinv.T)
            self.sigma_z = np.dot(Rotinv, self.sigma_z).dot(Rotinv.T)
            self.E_zz = self.N * self.sigma_z + np.dot(self.means_z.T, self.means_z) 
            self.Lqz += -2 * det  

            # Update W
            for i in range(0, self.s):
                self.means_w[i] = np.dot(self.means_w[i], Rot)
                self.sigma_w[i] = np.dot(Rot.T, self.sigma_w[i]).dot(Rot)
                self.E_WW[i] = self.d[i] * self.sigma_w[i] + \
                    np.dot(self.means_w[i].T, self.means_w[i])
                self.Lqw[i] += 2 * det 
        else:
            self.DoRotation = False
            print('Rotation stopped')       

    def Er(self, r):
        
        """ 
        Evaluates the (negative) cost function value wrt the 
        transformation matrix R used in the generic 
        optimization routine.

        Parameters
        ----------
        r : array-like
            Flatten transformation matrix R.                   
        
        """ 
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
        
        """ 
        Evaluates the (negative) gradient of the cost function Er().

        Parameters
        ----------
        r : array-like
            Flatten transformation matrix R.                   
        
        """
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

        """ 
        Shut down irrelevant/noisy latent components.                   
        
        """
        cols_rm = np.ones(self.k, dtype=bool)
        colMeans_Z = np.mean(self.means_z ** 2, axis=0)         
        if any(colMeans_Z < 1e-6):
            cols_rm[colMeans_Z < 1e-6] = False                
            self.means_z = self.means_z[:,cols_rm]
            self.sigma_z = self.sigma_z[:,cols_rm]
            self.sigma_z = self.sigma_z[cols_rm,:]
            self.E_zz = self.E_zz[:,cols_rm]
            self.E_zz = self.E_zz[cols_rm,:]
            self.k = self.means_z.shape[1]

            for i in range(0, self.s):
                self.means_w[i] = self.means_w[i][:,cols_rm]
                self.sigma_w[i] = self.sigma_w[i][:,cols_rm]
                self.sigma_w[i] = self.sigma_w[i][cols_rm,:]
                self.E_WW[i] = self.E_WW[i][:,cols_rm]
                self.E_WW[i] = self.E_WW[i][cols_rm,:]
                self.E_alpha[i] = self.E_alpha[i][cols_rm]        
