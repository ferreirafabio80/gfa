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

    params = {'num_sources': 2,
                      'K': 6, 'scenario': 'complete'}
    X = [np.random.rand(100,10) for _ in range(params['num_sources'])]

    return X, params