import numpy as np

def F(x):
    return (3*x - 1)**(1/3)

def punto_fijo(initial_guess: float, tolerance: float, max_iterations: int) -> (float, int):
    """

    Args:
        initial_guess: initial guess
        tolerance: tolerance
        max_iterations: maximum number of iterations
    
    Returns:
        (float, int): (solution, number of iterations)
    """
    x_prev = initial_guess
    for i in range(max_iterations):
        x_next = F(x_prev)
        if abs(x_next - x_prev) < tolerance:
            return x_next, i + 1
        x_prev = x_next
    return None, max_iterations