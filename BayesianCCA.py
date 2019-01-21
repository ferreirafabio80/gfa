import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma, wishart 

class BayesianCCA(object):

    def __init__(self, X1, X2, a_alpha=10e-3, b_alpha=10e-3, 
                 a_tau=10e-3, b_tau=10e-3, beta=10e-3):

        d1 = X1.shape[1]  # number of dimensions of X1
        d2 = X2.shape[1]  # number of dimensions of X2
        m = np.min(d1,d2) # number of different models 
        N = X1.shape[0]   # number of data points
