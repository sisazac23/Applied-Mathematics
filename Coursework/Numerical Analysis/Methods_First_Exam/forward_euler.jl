using LinearAlgebra

function forward_euler(A, y0, t, h)
    """
    Solve y' = Ay with y(0) = y0 by forward Euler method
    
    Args:
        A: matrix
        y0: initial condition for y
        t: 1D array of t values where we approximate y values. Time step
            at each iteration is given by t[n+1] - t[n].
        h: time step
        
    Returns:
        y: 2D array
            Approximation of the solution y(t) computed at each time in t.
    """
    
    y = zeros(eltype(y0), length(y0), length(t))
    y[:,1] = y0
    for i in 2:length(t)
        y[:,i] = y[:,i-1] + h * A * y[:,i-1]
    end
    return y
end
