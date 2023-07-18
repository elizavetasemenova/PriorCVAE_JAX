from abc import ABC, abstractmethod
import wandb

from .tests import mean_bootstrap_interval_contains_zero, norm_of_kernel_diff
from .visualizations import plot_vae_realizations, plot_kernel, plot_empirical_covariance

class AbstractTestRunner(ABC):
    def __init__(self, kernel, kernel_name, x):
        self.kernel = kernel
        self.kernel_matrix = kernel(x, x)
        self.kernel_name = kernel_name
        self.x = x

    @abstractmethod
    def run_tests(self, samples):
        pass

    @abstractmethod
    def run_visualizations(self, samples):
        pass

class SquaredExponentialTestRunner(AbstractTestRunner):
    def __init__(self, kernel, kernel_name, x):
        super().__init__(kernel, kernel_name, x)
        self.tests = [
            mean_bootstrap_interval_contains_zero,
            norm_of_kernel_diff,
        ]

        self.visualizations = [
            plot_empirical_covariance,
            plot_kernel,
            plot_vae_realizations,
        ]

    def run_tests(self, samples):
        for test in self.tests:
            result = test(samples=samples, kernel=self.kernel_matrix)
            wandb.run.summary[test.__name__] = result

    def run_visualizations(self, samples):
        for visualization in self.visualizations:
            visualization(samples=samples, kernel=self.kernel_matrix, x=self.x, kernel_name=self.kernel_name)


class MaternTestRunner(AbstractTestRunner):
    def __init__(self, kernel, kernel_name, x):
        super().__init__(kernel, kernel_name, x)

    def run_tests(self, samples):
        pass

    def run_visualizations(self, samples):
        pass
