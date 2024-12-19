import matplotlib.pyplot as plt


def plot_gaussians_vs_fourier_sum(x, y, fourier_sum, Z):
    """this function plots a sum over 2D gaussians, and their fourier sum."""
    contour = plt.contourf(x, y, fourier_sum, levels=35, cmap='jet')
    plt.colorbar(contour)
    plt.xlabel('x'), plt.ylabel('y')
    plt.title('Fourier series')
    plt.show()
    
    contour = plt.contourf(x, y, Z, levels=30, cmap='jet')
    plt.colorbar(contour)
    plt.xlabel('x'), plt.ylabel('y')
    plt.title('Original gaussians')
    plt.show()