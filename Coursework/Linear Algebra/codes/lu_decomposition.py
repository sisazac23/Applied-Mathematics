import numpy as np

def lu_decomposition(A: np.ndarray):
    """
    LU decomposition of a square matrix A using Doolittle's algorithm.

    Args:
        A: A square matrix. 
            
    Returns:
        L: A lower triangular matrix.
        U: An upper triangular matrix.
    """
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        for k in range(i, n):
            sum = 0
            for j in range(i):
                sum += L[i][j] * U[j][k]
            U[i][k] = A[i][k] - sum
        for k in range(i, n):
            if i == k:
                L[i][i] = 1
            else:
                sum = 0
                for j in range(i):
                    sum += L[k][j] * U[j][i]
                L[k][i] = (A[k][i] - sum) / U[i][i]
    return L, U