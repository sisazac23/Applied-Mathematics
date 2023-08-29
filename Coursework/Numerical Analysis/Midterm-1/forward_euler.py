import numpy as np

def forward_euler(A, y0, t, h):
    """
    Solve y' = Ay with y(0) = y0 by forward Euler method

    Args:
        A: matrix
        y0: initial condition for y
        t: 1D NumPy array of t values where we approximate y values. Time step
            at each iteration is given by t[n+1] - t[n].
        h: time step
    
    Returns:
        y: 2D NumPy array
            Approximation of the solution y(t) computed at each time in t.
    """
    
    y = np.zeros((len(y0), len(t)))
    y[:,0] = y0
    for i in range(1, len(t)):
        y[:,i] = y[:,i-1] + h*A@y[:,i-1]
    return y