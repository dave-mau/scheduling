import matplotlib.pyplot as plt
import numpy as np


def plot_gamma_histogram(shape, scale, num_samples=10000, num_bins=50):
    # Generate random samples from the gamma distribution
    samples = np.random.gamma(shape, scale, num_samples) * 4.0 + 40.0

    # Create the histogram
    count, bins, _ = plt.hist(samples, bins=num_bins, density=True, alpha=0.7, label="Sampled Data")

    plt.xlabel("X")
    plt.ylabel("Density")
    plt.title("Gamma Distribution: Histogram and PDF")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_normal_histogram(mu, std, num_samples=10000, num_bins=50):
    # Generate random samples from the gamma distribution
    samples = np.random.normal(mu, std, num_samples)

    # Create the histogram
    count, bins, _ = plt.hist(samples, bins=num_bins, density=True, alpha=0.7, label="Sampled Data")

    plt.xlabel("X")
    plt.ylabel("Density")
    plt.title("Gamma Distribution: Histogram and PDF")
    plt.grid(True)
    plt.legend()
    plt.show()


# Example usage:
shape_param = 7.5
scale_param = 1.0
plot_gamma_histogram(shape_param, scale_param)
# plot_normal_histogram(0, 4)
