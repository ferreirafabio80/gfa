import scipy.special as sp
import scipy.special as special
class BayesianPCA(object):
    
    def __init__(self, d, N, a_alpha=10e-3, b_alpha=10e-3, a_tau=10e-3, b_tau=10e-3, beta=10e-3):
        """
        """
        self.d = d # number of dimensions
        q = d
        self.q = q
        self.N = N # number of data points
        
        # Hyperparameters
        self.a_alpha = a_alpha
        self.b_alpha = b_alpha
        self.a_tau = a_tau
        self.b_tau = b_tau
        self.beta = beta

        # Variational parameters
        self.means_z = np.random.randn(q, N) # called x in bishop99
        self.sigma_z = np.random.randn(q, q)
        self.mean_mu = np.random.randn(d, 1)
        self.sigma_mu = np.random.randn(d, d)
        self.means_w = np.random.randn(d, q)
        self.sigma_w = np.random.randn(q, q)
        self.a_alpha_tilde = self.a_alpha + d/2.0
        self.bs_alpha_tilde = np.abs(np.random.randn(q, 1))
        self.a_tau_tilde = self.a_tau + (N*d)/2.0
        self.b_tau_tilde = np.abs(np.random.randn(1))
    
    def __update_z(self, X):
        self.sigma_z = np.linalg.inv(np.identity(self.d) + (self.a_tau_tilde/self.b_tau_tilde)*( np.trace(self.sigma_w) + dot(self.means_w.T,self.means_w)))
        self.means_z = (self.a_tau_tilde/self.b_tau_tilde)*dot(dot(self.sigma_z,self.means_w.T) , (X - self.mean_mu))
    
    def __update_mu(self,X):
        E_tau = self.a_tau_tilde / self.b_tau_tilde
        E_W = self.means_w
        S = 0
        for n in range(0,self.N):
            t_n = np.reshape(X.T[n],(self.d,1)) 
            x_n = np.reshape(self.means_z.T[n],(self.q,1)) 
            S += t_n - dot(E_W,x_n)
        self.mean_mu = E_tau*dot(self.sigma_mu,S)
        self.sigma_mu = np.identity(self.d) / (self.beta + self.N*(self.a_tau_tilde/self.b_tau_tilde)) 
        
    def __update_w(self, X):
        self.sigma_w = np.linalg.inv( diagflat(self.a_alpha_tilde/self.bs_alpha_tilde) + (self.a_tau_tilde/self.b_tau_tilde)*(self.N*self.sigma_z + dot(self.means_z,self.means_z.T) ) )
        E_tau = self.a_tau_tilde / self.b_tau_tilde
        for k in range(0,self.d):
            S = 0
            mu_k = self.mean_mu[k]
            for n in range(0,self.N):
                t_n_k = X.T[n][k]
                x_n = np.reshape(self.means_z.T[n],(self.q,1))
                S += x_n*(t_n_k - mu_k)
            self.means_w[k] = np.reshape( E_tau* dot(self.sigma_w,S) , self.q )
    
    
    def __update_alpha(self):
        for i in range(0 , self.q):
            self.bs_alpha_tilde[i] = self.b_alpha + (np.trace(self.sigma_w)+dot(self.means_w.T[i].T,self.means_w.T[i])) / 2
         
        
    def __update_tau(self, X):
        A = 0
        E_W = self.means_w
        for i in range(0,self.N):
            t_n = np.reshape(X.T[i],(self.d,1))
            x_n = np.reshape(self.means_z.T[i],(self.q,1))
            A += dot(t_n.T,t_n) + np.trace(self.sigma_mu) + dot(self.mean_mu.T,self.mean_mu)
            A += np.trace( dot( np.trace(self.sigma_w) + dot(E_W.T,E_W), self.sigma_z + dot(x_n,x_n.T)) )
            A += 2*dot(dot(self.mean_mu.T,self.means_w),x_n)
            A += -2* dot(dot(t_n.T,self.means_w),x_n) - 2*dot(t_n.T,self.mean_mu)
        self.b_tau_tilde = self.b_tau + 0.5*A

    def L(self, X):
        
        L = 0
        ###Terms from expectations###
        #N(X_n|Z_n)
        L += -self.N*self.d/2 * (special.digamma(self.a_tau_tilde) - np.log(self.b_tau_tilde)) - self.d/2 *np.log(2*np.pi)
        S = 0
        for n in range(0,self.N):
            S += dot(X[:,n].T, X[:,n])
            S += dot(dot(X[:,n].T,self.means_w),self.means_z[:,n]) 
            S += dot(X[:,n].T, self.mean_mu)[0]
            S -= dot(dot(self.means_w.T,self.means_z[:,n].T) ,X[:,n])
            S -= dot(dot(self.means_w.T,self.means_z[:,n].T) ,self.mean_mu)    
            S += dot( self.mean_mu.T,X[:,n])[0]
            S -= dot(dot(self.mean_mu.T,self.means_w) ,self.means_z[:,n]) 
            S += np.trace(self.sigma_mu) + dot( self.mean_mu.T,self.mean_mu)[0]
            S += dot(self.mean_mu.T, X[:,n].T)[0]
            S -= dot (dot (self.mean_mu.T, self.means_w), self.means_z[:,n])[0]
            S += np.trace(self.sigma_mu) + dot(self.mean_mu.T, self.mean_mu)[0][0]
        L +=  self.N/2 *(self.a_tau_tilde/self.b_tau_tilde)*S
        
        #sum ln N(z_n)
        L += - self.N / 2 * np.trace(self.sigma_z) - self.N*self.d/2 *np.log(2*np.pi)
        for n in range(0,self.N):
            L += - 1/2 * dot(self.means_z.T[n].T, self.means_z.T[n])
        
        #sum ln(W|a)
        L += self.q * self.d /2 * (- np.log(2*np.pi) )
        for i in range(0, self.q):
            L+= -self.q /2 * ((special.digamma(self.a_alpha_tilde) - log(self.bs_alpha_tilde[i][0]) ) + self.a_alpha_tilde / self.bs_alpha_tilde[i][0]) /   (- 1/2) *  (np.trace(self.sigma_w) + dot(self.means_w.T[i].T, self.means_w.T[i]))
        
        #sum ln (Ga(a_i))
        L += self.q * (- log(special.gamma(self.a_alpha)) + self.a_alpha * log(self.b_alpha) )
        for i in range(0, self.q):
            L +=  -log(self.bs_alpha_tilde[i][0]) + special.digamma(self.a_alpha_tilde)- self.b_alpha * (self.a_alpha_tilde / self.bs_alpha_tilde[i][0])
        
        #ln(N(\mu))
        L += self.d/2 * (np.log(self.beta) - np.log(2 * np.pi) ) - self.beta/2*(np.trace(self.sigma_mu) + dot(self.mean_mu.T, self.mean_mu)[0][0])
        
        #ln(Ga(\tau))
        L +=  -log(self.b_tau_tilde[0]) + special.digamma(self.a_tau_tilde) - (self.a_tau_tilde / self.b_tau_tilde[0])

        ###Terms from entropies
        #H[Q(Z)]
        L += self.N*( self.d/2*(1+ np.log(2*np.pi)) + 1/2 * log(linalg.det(self.sigma_z)))
        
        #H[Q(\mu)]
        L += (0.5)*log(linalg.det(self.sigma_mu))  + self.d/2*(1+ np.log(2*np.pi))
        
        #H[Q(W)]
        L += self.d*(self.d /2 *(1+ np.log(2*np.pi)) + 1/2*log(linalg.det(self.sigma_w)) )
        
        #H[Q(\alpha)]
        L += self.q * (self.a_alpha_tilde + log(special.gamma(self.a_alpha_tilde)) + (1-self.a_alpha_tilde)*special.digamma(self.a_alpha_tilde)) 
        for i in range(0,self.d):
            L += -log(self.bs_alpha_tilde[i][0])
      
        #H[Q(\tau)]
        L += self.a_tau_tilde - log(self.b_tau_tilde[0]) + (1-self.a_tau_tilde) * special.digamma(self.a_tau_tilde)
        ## the term Gamma(a_tau_tilde) is inf so we ignore it...
        
        return L
    
    def fit(self, X,iterations = 1000, threshold = 1):
        self.__update_tau(X)
        L_previous = -1000000000
        L_new = 0
        i = 0
        for i in range(iterations):
            self.__update_tau(X)
            self.__update_mu(X)
            self.__update_alpha()            
            self.__update_w(X)
            self.__update_z(X)  
            if i % 100 == 1 :
                print "Iterations: " , i
                print "Lower Bound Value : " , self.L(X)
            i += 1
            L_previous = L_new
            L_new = self.L(X)
            if abs(L_new - L_previous) < threshold:
                break