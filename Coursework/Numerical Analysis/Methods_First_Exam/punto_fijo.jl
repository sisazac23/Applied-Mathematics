# Define the function F
function F(x)
    return (3*x - 1)^(1/3)
end

# Define the punto_fijo function
function punto_fijo(initial_guess::Float64, tolerance::Float64, max_iterations::Int)
    """
    Args:
        initial_guess: initial guess
        tolerance: tolerance
        max_iterations: maximum number of iterations
    
    Returns:
        (Float64, Int): (solution, number of iterations)
    """
    x_prev = initial_guess
    for i in 1:max_iterations
        x_next = F(x_prev)
        if abs(x_next - x_prev) < tolerance
            return x_next, i
        end
        x_prev = x_next
    end
    return nothing, max_iterations
end