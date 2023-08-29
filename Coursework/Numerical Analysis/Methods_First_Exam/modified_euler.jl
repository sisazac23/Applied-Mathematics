using LinearAlgebra

function modified_euler(A, y0, t, h)
    """
    Solve y' = Ay with y(0) = y0 using the modified Euler method
    
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
    for i in 2:length(t)
        y[i, :] = y[i-1, :] + h/2 * (A * y[i-1, :] + A * (y[i-1, :] + h * A * y[i-1, :]))
    end
    return y
end
