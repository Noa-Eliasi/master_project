import functions as fnc
import plots as pts
import numpy as np

## calculating the fourier sum and the gaussians on the lattice, and plotting them ##
x, y = np.meshgrid(np.linspace(-4, 4, 300), np.linspace(-4, 4, 300))
sigma = 0.18
n, m = 2, 2

Z = fnc.gaussian(x, y, 1, n, m, sigma)
fourier_sum = fnc.fourier_series(1, sigma, n, m, x, y)
pts.plot_gaussians_vs_fourier_sum(x, y, fourier_sum, Z)
