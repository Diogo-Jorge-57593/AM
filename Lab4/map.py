import math
import numpy as np

def mapFeature(X1, X2, degree=6):
    '''
    Expands the 2 input features X1 and X2 to polynomial features
    
    Returns a new feature array with the expansion :
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, ...

    Inputs X1, X2 must be Numpy 2D arrays of the same size
    '''

    m = X1.shape[0]
    out = np.ones((m,1))
    
    for i in range (1,degree+1):
        for j in range(0,i+1):
            v = (X1**(i-j))*(X2**j)
            out = np.c_[out, v]
    return out

