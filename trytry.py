import numpy as np
from scipy.optimize import fmin_ncg, fmin_cg, approx_fprime, minimize
from tqdm import tqdm
import matplotlib.pyplot as plt

# constants
a = 1
sigma = 0.15
N, M = 3, 3
alpha = 0.1 * np.pi / 180
mu = 0.1
lamda = 0.05

nx, ny = 50, 50
x0, xf = 0, 2
y0, yf = 0, 2
x_lin, y_lin = (np.linspace(x0, xf, nx), np.linspace(y0, yf, ny))
x, y = np.meshgrid(x_lin, y_lin)


def gradient_func(x):
    eps = np.sqrt(np.finfo(float).eps)
    return approx_fprime(x, lagrangian, eps)



def fourier_series(a, sigma, N, M, x, y):
    """This function creates a Fourier series for a sum over 2D Gaussians."""
    b_1 = np.array([2 * np.pi / a, -2 * np.pi / (a * np.sqrt(3))])
    b_2 = np.array([0, 4 * np.pi / (a * np.sqrt(3))])
    f = np.zeros_like(x)
    for n in range(-N, N+1):
        for m in range(-M, M+1):
            G = n * b_1 + m * b_2
            coefficient = 4 * np.pi * sigma ** 2 * np.exp(-2 * np.pi ** 2 * sigma ** 2 * np.linalg.norm(G) ** 2) / (a ** 2 * np.sqrt(3))
            f += coefficient * np.cos(x * G[0] + y * G[1])

    f -= np.mean(f)
    f = f / (np.max(f) - np.min(f))  # Normalize
    return f


def lagrangian(u_flat):
    u = u_flat.reshape((nx, ny, 2))
    u0 = np.stack((-alpha * y, alpha * x), axis=-1)
    total_u = u0 + u
    du_dx = np.gradient(u, axis=1) / (x_lin[1] - x_lin[0])
    du_dy = np.gradient(u, axis=0) / (y_lin[1] - y_lin[0])
    divergence_u = du_dx[:, :, 0] + du_dy[:, :, 1]

    psi_values = fourier_series(a, sigma, N, M, total_u[:, :, 0], total_u[:, :, 1])
    jacobian_u = np.stack([du_dx, du_dy], axis=-1)
    Q_alpha = np.array([[0, -alpha], [alpha, 0]])
    norm_term = np.linalg.norm(Q_alpha + jacobian_u, axis=(-2, -1))

    integral_psi = np.mean(psi_values)
    integral_norm = mu * np.mean(norm_term)
    integral_div = lamda * np.mean(divergence_u ** 2)
    return integral_psi + integral_norm + integral_div




initial_guess = np.zeros(nx * ny * 2)

# setup tqdm progress bar
def callback(xk):
    pbar.update(1)

n_iterations = 10
pbar = tqdm(total=n_iterations, desc="Optimizing")


# ## using the optimize(NCG) method
# # minimizing the lagrangian
# solution = minimize(lagrangian, initial_guess, method='Newton-CG', jac=gradient_func, callback=callback)
# pbar.close()
# # take result back to grid shape
# delta_u_optimal = solution.x.reshape((nx, ny, 2))

## using the fnmin_cg method
# minimizing the lagrangian
solution = fmin_cg(lagrangian, initial_guess, callback=callback)
pbar.close()
# # take result back to grid shape
delta_u_optimal = solution.reshape((nx, ny, 2)) 

print("Optimal delta u found:")
print(delta_u_optimal)





plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.contourf(x, y, delta_u_optimal[:, :, 0], levels=50, cmap='viridis')
plt.colorbar()
plt.title('u_x')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(1, 2, 2)
plt.contourf(x, y, delta_u_optimal[:, :, 1], levels=50, cmap='viridis')
plt.colorbar()
plt.title('u_y')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()
