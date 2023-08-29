import numpy as np

def cholesky(A: np.ndarray) -> np.ndarray:
    '''    
    Cholesky decomposition of a symmetric positive definite matrix A

    parameters:
        A: symmetric positive definite matrix
        
    returns: 
        lower triangular matrix L
    '''


    n = len(A)
    L = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            tmp_sum = 0
            for k in range(j):
                tmp_sum += L[i,k] * L[j,k]
            if (i == j):
                L[i,j] = np.sqrt(A[i,i] - tmp_sum)
            else:
                L[i,j] = (A[i,j] - tmp_sum) / L[j,j]
    return L
