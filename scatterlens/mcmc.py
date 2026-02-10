import torch
import numpy as np
from emcee import EnsembleSampler
from typing import Callable
from scatterlens.emulator import Emulator

# Part of this code is copied from Lars.
# Source: https://github.com/ljabbo/kids-persistent-homology

class MCMC:
    def __init__(
            self,
            emulator: Emulator,
            data_vector: np.ndarray,
            covariance_matrix: np.ndarray,
            likelihood_type: str = "Sellentin_Heavens",
            param_ranges: dict[str, list[float, float]] = None,
            n_simulations: int = 217,
    ):
        """Initialize the MCMC class."""
        if torch.is_tensor(data_vector):
            data_vector = data_vector.numpy()

        if torch.is_tensor(covariance_matrix):
            covariance_matrix = covariance_matrix.numpy()

        self.emulator = emulator
        self.dv = data_vector

        try:
            np.linalg.cholesky(covariance_matrix)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix is not positive definite.")

        self.cov = covariance_matrix
        self.inv_cov = np.linalg.inv(covariance_matrix)
        self.likelihood_type = likelihood_type
        self.param_ranges = param_ranges
        self.n_dim = len(param_ranges.keys()) # number of free parameters being sampled
        self.n_simulations = n_simulations # number of simulations for covariance estimation
        self.sampler = None

    def get_log_likelihood_function(self) -> Callable:
        match self.likelihood_type:
            case "Sellentin_Heavens":
                return self._Sellentin_Heavens_log_likelihood
            case "Gaussian":
                return self._Gaussian_log_likelihood
            case _:
                raise ValueError(f"Unknown likelihood type: {self.likelihood_type}")


    def _log_prior(self, cosm_params):
        """Log flat prior for the cosmological parameters."""
        # TODO support Gaussian priors
        for param, value in zip(self.param_ranges.keys(), cosm_params):
            if value < self.param_ranges[param][0] or value > self.param_ranges[param][1]:
                return -np.inf
        return 0.0


    def _Sellentin_Heavens_log_likelihood(self, cosm_params):
        chi2 = self.chi2(cosm_params)
        posterior = -0.5 * self.n_simulations * np.log(1 + chi2 / self.n_simulations)
        posterior += self._log_prior(cosm_params)
        return posterior

    def _Hartlap(self, cosm_params):
        pass

    def _Gaussian_log_likelihood(self, cosm_params):
        return -0.5 * np.log * (self.chi2(cosm_params)) + self._log_prior(cosm_params)

    def chi2(self, cosm_params):
        if not isinstance(cosm_params, np.ndarray):
            cosm_params = np.array(cosm_params)

        if cosm_params.ndim == 1:
            cosm_params = cosm_params.reshape(1, -1)

        pred = self.emulator.predict(cosm_params).squeeze()
        return (self.dv - pred).T @ self.inv_cov @ (self.dv - pred)

    def _get_random_walk(self, n_walkers: int):
        """Generate a random walk for the MCMC."""
        return np.random.uniform(
            low=[self.param_ranges[param][0] for param in self.param_ranges],
            high=[self.param_ranges[param][1] for param in self.param_ranges],
            size=(n_walkers, self.n_dim),
        )

    def run_mcmc(
            self,
            n_walkers: int=100,
            n_steps: int=500,
            n_burn_in_steps: int=100,
            **kwargs,
    ):
        """Run the MCMC sampler.

        Args:
            n_walkers: Number of walkers in the MCMC sampler.
            n_steps: Number of steps in the MCMC sampler.
            n_burn_in_steps: Number of burn-in steps to discard.
            kwargs: Additional keyword arguments for the
                emcee.EnsembleSampler.sample method.
        """
        p0 = self._get_random_walk(n_walkers=n_walkers)
        log_likelihood = self.get_log_likelihood_function()

        sampler = EnsembleSampler(n_walkers, self.n_dim, log_likelihood)
        state = sampler.run_mcmc(p0, n_burn_in_steps)
        sampler.reset()
        sampler.run_mcmc(state, n_steps, **kwargs)

        self.sampler = sampler


    def get_chain(self, flat=True):
        """Get the chain of samples from the MCMC sampler."""
        if self.sampler is None:
            raise ValueError("MCMC sampler has not been run yet.")
        else:
            return self.sampler.get_chain(flat=flat)


def Hartlap_factor(n_simulations: int, dv_length: int) -> float:
    """Calculate the Hartlap factor for covariance matrix correction."""
    return  (n_simulations - 1) / (n_simulations - dv_length - 2)