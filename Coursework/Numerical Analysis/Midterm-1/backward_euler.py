import numpy as np

def backward_euler(A,y0,t,h):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = np.linalg.solve(np.eye(len(A)) - h*A, y[i-1])
    return y
