import numpy as np


def order_of_convergence(x: list, y: list):
    """
    Calculates the order of convergence between two sequences.
    """

    x = np.array(x)
    y = np.array(y)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    if len(x) < 2:
        raise ValueError("x and y must have at least two elements")

    # make fit
    log_x = np.log(x)
    log_y = np.log(y)
    coeffs = np.polyfit(log_x, log_y, 1)
    order = -coeffs[0]

    return order
