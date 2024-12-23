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
    """This function creates a Fourier series for a sum over 2D Gaussians."""
    # reciprocal lattice vectors
    b_1 = np.array([2 * np.pi / a, -2 * np.pi / (a * np.sqrt(3))])
    b_2 = np.array([0, 4 * np.pi / (a * np.sqrt(3))])
    f = np.zeros_like(x)
    for n in range(-N, N+1):
        for m in range(-M, M+1):
            G = n * b_1 + m * b_2
            coefficient = 4 * np.pi * sigma ** 2 * np.exp(-2 * np.pi ** 2 * sigma ** 2 * np.linalg.norm(G) ** 2) / (a ** 2 * np.sqrt(3))
            f += coefficient * np.cos(x * G[0] + y * G[1])
    f -= np.mean(f)
    f = f / (np.max(f) - np.min(f))
    return f


# def derivatives(delta_ux, delta_uy, dx, dy):
#     dxux = (delta_ux[2:, 1:-1] - delta_ux[:-2, 1:-1]) / dx
#     dxuy = (delta_uy[2:, 1:-1] - delta_uy[:-2, 1:-1]) / dx
#     dyux = (delta_ux[1:-1, 2:] - delta_ux[1:-1, :-2]) / dy
#     dyuy = (delta_uy[1:-1, 2:] - delta_uy[1:-1, :-2]) / dy
#     return dxux, dxuy, dyux, dyuy


# def lagrangian_density(delta_ux, delta_uy, a, r, mu, lamda, dx, dy):
#     Q_a = np.array([[0, -a], [a, 0]])
#     u_0 = np.matmul(Q_a, r) 

#     x, y = np.meshgrid(np.linspace(-4, 4, 300), np.linspace(-4, 4, 300))
#     # dx, dy = 

#     dxux, dxuy, dyux, dyuy = derivatives(delta_ux, delta_uy, dx, dy)
#     u_x, u_y = delta_ux + u_0[0], delta_uy + u_0[1]
#     grad_delta_u = np.array([[dxux, dyux], [dxuy, dyuy]])

#     expr_1 = fourier_series(1, 0.18, 3, 3, u_x, u_y)
#     expr_2 = mu * np.linalg.norm(Q_a + grad_delta_u)
#     expr_3 = lamda * (dxux + dyuy) ** 2

#     return expr_1 + expr_2 + expr_3