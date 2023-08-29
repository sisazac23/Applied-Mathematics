# define forward euler method for second order ode's

import numpy as np
import matplotlib.pyplot as plt

def forward_euler(f, x0, t):
    """
    Solve x' = f(x,t) with x(0) = x0 by forward Euler method
    
    Parameters
    ----------
    f : function
        Right-hand side of the differential equation
    x0 : number
        Initial condition for x
    t : array
        1D NumPy array of t values where we approximate x values. Time step
        at each iteration is given by t[n+1] - t[n].
        
    Returns
    -------
    x : 1D NumPy array
        Approximation of the solution x(t) computed at each time in t.
    """
    # Initialize x array
    x = np.zeros(len(t))
    
    # Set initial condition
    x[0] = x0
    
    # Time step
    dt = t[1] - t[0]
    
    # Iterate through time steps
    for n in range(0, len(t)-1):
        x[n+1] = x[n] + dt*f(x[n], t[n])
        
    return x