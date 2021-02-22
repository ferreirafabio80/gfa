#Author: Fabio S. Ferreira (fabio.ferreira.16@ucl.ac.uk)
#Date: 22 February 2021
import pytest
import numpy as np
from models import GFA_OriginalModel, GFA_DiagonalNoiseModel

def test_gfaori_smoke_test():

    X, params = generate_data()
    gfa = GFA_OriginalModel(X, params)
    assert gfa is not None

def test_gfanew_smoke_test():

    X, params = generate_data()
    gfa = GFA_DiagonalNoiseModel(X, params)
    assert gfa is not None  

def generate_data():

    params = {'num_groups': 2,
                      'K': 6, 'scenario': 'complete'}
    X = [np.random.rand(100,10) for _ in range(params['num_groups'])]

    return X, params