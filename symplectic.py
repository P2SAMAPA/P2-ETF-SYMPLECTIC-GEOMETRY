import numpy as np
from scipy.stats import entropy

def phase_space_embedding(returns, momentum_window=5):
    """
    Construct phase space: position = return, momentum = return - lagged return.
    """
    # Ensure returns is a numpy array
    returns = np.asarray(returns).flatten()
    momentum = returns - np.roll(returns, momentum_window)
    momentum[:momentum_window] = 0
    return returns, momentum

def poincare_section(returns, momentum, section_value=0.0, epsilon=0.01):
    """
    Take Poincaré section at momentum = section_value (± epsilon).
    Returns the values of returns at crossings.
    """
    crossings = []
    for i in range(1, len(momentum)):
        if abs(momentum[i] - section_value) < epsilon:
            crossings.append(returns[i])
    return np.array(crossings)

def first_return_map(crossings):
    """
    Compute first return map: y_{n+1} = f(y_n) from Poincaré section crossings.
    """
    if len(crossings) < 2:
        return np.array([]), np.array([])
    # Remove any NaN
    crossings = crossings[~np.isnan(crossings)]
    if len(crossings) < 2:
        return np.array([]), np.array([])
    return crossings[:-1], crossings[1:]

def return_map_entropy(returns, momentum, n_bins=20):
    """
    Compute entropy of the Poincaré section return map.
    High entropy = chaotic, low entropy = regular/periodic.
    """
    returns = np.asarray(returns).flatten()
    momentum = np.asarray(momentum).flatten()
    # Remove NaN/inf
    mask = ~(np.isnan(returns) | np.isinf(returns) | np.isnan(momentum) | np.isinf(momentum))
    returns = returns[mask]
    momentum = momentum[mask]
    if len(returns) < 10:
        return 0.0
    crossings = poincare_section(returns, momentum, section_value=0.0, epsilon=0.01)
    if len(crossings) < 5:
        return 0.0
    x, y = first_return_map(crossings)
    if len(x) == 0:
        return 0.0
    # Remove any remaining NaN/inf
    mask2 = ~(np.isnan(x) | np.isinf(x) | np.isnan(y) | np.isinf(y))
    x = x[mask2]
    y = y[mask2]
    if len(x) < 2:
        return 0.0
    # Compute 2D histogram
    try:
        hist, x_edges, y_edges = np.histogram2d(x, y, bins=n_bins)
    except:
        return 0.0
    hist_flat = hist.flatten()
    p = hist_flat / (hist_flat.sum() + 1e-12)
    ent = entropy(p[p > 0])
    return ent

def symplectic_score(returns, n_bins=20, momentum_window=5):
    """
    Compute Poincaré section entropy for a single ETF return series.
    """
    if len(returns) < momentum_window + 10:
        return 0.0
    pos, mom = phase_space_embedding(returns, momentum_window)
    ent = return_map_entropy(pos, mom, n_bins)
    return float(ent)
