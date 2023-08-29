using LinearAlgebra

function rk4(A, y0, t, h)
    """
    Solve y' = Ay with y(0) = y0 using the fourth-order Runge-Kutta method
    
    Args:
        A: matrix
        y0: initial condition for y
        t: 1D array of t values where we approximate y values. Time step
            at each iteration is given by t[n] - t[n-1].
        h: time step
        
    Returns:
        y: 2D array
            Approximation of the solution y(t) computed at each time in t.
    """
    
    y = zeros(eltype(y0), length(t), length(y0))
    y[1, :] = y0
    for i in 1:(length(t) - 1)
        k1 = h * A * y[i, :]
        k2 = h * A * (y[i, :] + k1 / 2)
        k3 = h * A * (y[i, :] + k2 / 2)
        k4 = h * A * (y[i, :] + k3)
        y[i+1, :] = y[i, :] + (k1 + 2*k2 + 2*k3 + k4) / 6
    end
    return y
end
