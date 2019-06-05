import numpy as np

def add(A, B):
    '''
    A - np matrix
    B - np matrix
    '''
    return_dict = {"output": A + B,
                   "gradient_A": np.ones(A.shape),
                   "gradient_B": np.ones(B.shape)}
    return return_dict


def minus(A, B):
    '''
    A - np matrix
    B - np matrix
    '''
    return_dict = {"output": A - B,
                   "gradient_A": np.ones(A.shape),
                   "gradient_B": -np.ones(B.shape)}
    return return_dict


def multiply(A, B):
    '''
    A - np matrix
    B - np matrix
    '''
    return_dict = {"output": A * B,
                   "gradient_A": B,
                   "gradient_B": A.T}
    return return_dict


def maximum(A):
    '''
    A - np matrix
    '''
    maximum = np.max(A)
    gradient = (A == maximum).astype(int)
    return_dict = {"output": maximum,
                   "gradient": gradient}
    return return_dict


def minimum(A):
    '''
    A - np matrix
    '''
    minimum = np.min(A)
    gradient = (A == minimum).astype(int)
    return_dict = {"output": minimum,
                   "gradient": gradient}
    return return_dict
