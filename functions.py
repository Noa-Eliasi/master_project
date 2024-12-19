import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt


def create_lattice(a, N, M):
    """this function creates a set of coordinates that for a hexagonal lattice."""
    # lattice vectors
    a_1 = np.array([a, 0])
    a_2 = np.array([a / 2, a * np.sqrt(3) / 2])
    mu_x, mu_y = [], []
    for n in range(N):
        for m in range(M):
            mu_x.append((n * a_1 + m * a_2)[0])
            mu_y.append((n * a_1 + m * a_2)[1])
    return mu_x, mu_y


def gaussian(x, y, a, n, m, sigma):
    """this function creates a pattern of a sum of 2D gaussians on a hexagonal lattice."""
    mu_x, mu_y = create_lattice(a, n, m)
    Z = np.zeros_like(x)
    for mx, my, in zip(mu_x, mu_y):
        Z += np.exp(-((x - mx) ** 2 + (y - my) ** 2) / (2 * sigma ** 2))
    return Z


def fourier_series(a, sigma, N, M, x, y):
    """this function creates a fourier series for a sum over 2D gaussians"""
    # reciprocal lattice vectors
    b_1 = np.array([2 * np.pi / a, -2 * np.pi / (a * np.sqrt(3))])
    b_2 = np.array([0, 4 * np.pi / (a * np.sqrt(3))])
    f = np.zeros_like(x)
    for n in range(N):
        for m in range(M):
            G = n * b_1 + m * b_2
            coefficient = 4 * np.pi * sigma ** 2 * np.exp(-2 * np.pi ** 2 * sigma ** 2 * np.linalg.norm(G) ** 2) / (a ** 2 * np.sqrt(3))
            f += coefficient * np.cos(x * G[0] + y * G[1])
    f = f - np.mean(f)
    f = f / (np.max(f) - np.min(f)) - np.min(f / (np.max(f) - np.min(f)))
    return f