import torch
import numpy as np
from emcee import EnsembleSampler
from typing import Callable

from scatterlens.model import Model
from scatterlens.utils import _tqdm_style

# Part of this code is copied from Lars.
# Source: https://github.com/ljabbo/kids-persistent-homology

class MCMC:
    def __init__(
            self,
            model: Model,
            data_vector: np.ndarray,
            covariance_matrix: np.ndarray,
            likelihood_type: str="Sellentin_Heavens",
            param_priors: dict[str, list[str, list[float, float | np.ndarray]]]=None,
            n_simulations: int=217,
    ):
        """Initialize the MCMC class."""
        if torch.is_tensor(data_vector):
            data_vector = data_vector.numpy()

        if torch.is_tensor(covariance_matrix):
            covariance_matrix = covariance_matrix.numpy()

        self.model = model
        self.dv = data_vector

        try:
            np.linalg.cholesky(covariance_matrix)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix is not positive definite.")

        self.cov = covariance_matrix
        self.inv_cov = np.linalg.inv(covariance_matrix)
        self.likelihood_type = likelihood_type
        self.param_priors = param_priors
        self.n_dim = len(param_priors.keys()) # number of free parameters being sampled
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


    def _log_prior(self, params):
        """Log prior of the parameters.

        Notes:
            Support flat priors, ["flat", (low, high)], and Gausssian priors,
            ["gaussian", (mean, std)].
        """
        # TODO: support correlated priors (i.e. photometric redshift)
        log_prior = 0.

        for param, value in zip(self.param_priors.keys(), params):
            prior_type = self.param_priors[param][0]

            if prior_type == "flat":
                low, high = self.param_priors[param][1]
                if value < low or value > high:
                    return -np.inf

            if prior_type == "gaussian":
                mean, cov = self.param_priors[param][1]
                if isinstance(cov, float):
                    log_prior += -0.5 * ((value - mean) / cov) ** 2
                else:
                    pass

        return log_prior


    def _Sellentin_Heavens_log_likelihood(self, params):
        # TODO: double check the formula for the likelihood
        chi2 = self.chi2(params)
        posterior = -0.5 * self.n_simulations * np.log(1 + chi2 / (self.n_simulations - 1))
        posterior += self._log_prior(params)
        return posterior

    def _Gaussian_log_likelihood(self, params):
        return -0.5 * (self.chi2(params)) + self._log_prior(params)

    def chi2(self, params):
        if not isinstance(params, np.ndarray):
            params = np.array(params)

        pred = self.model.predict(params).squeeze()
        return (self.dv - pred).T @ self.inv_cov @ (self.dv - pred)

    def _get_random_walk(self, n_walkers: int):
        """Generate a random walk for the MCMC."""
        # TODO: support correlated priors (i.e. photometric redshift)
        p0 = np.zeros((n_walkers, self.n_dim))

        for i, param in enumerate(self.param_priors.keys()):
            prior_type = self.param_priors[param][0]

            if prior_type == "flat":
                low, high = self.param_priors[param][1]
                p0[:, i] = np.random.uniform(low=low, high=high, size=n_walkers)

            if prior_type == "gaussian":
                mean, cov = self.param_priors[param][1]
                if isinstance(cov, float):
                    p0[:, i] = np.random.normal(loc=mean, scale=cov, size=n_walkers)
                else:
                    pass

        return p0


    def run_mcmc(
            self,
            n_walkers: int=100,
            n_steps: int=500,
            n_burn_in_steps: int=100,
            progress: bool=True,
            progress_kwargs=_tqdm_style,
            **kwargs,
    ):
        """Run the MCMC sampler.

        Args:
            n_walkers: Number of walkers in the MCMC sampler.
            n_steps: Number of steps in the MCMC sampler.
            n_burn_in_steps: Number of burn-in steps to discard.
            progress: Show a progress bar using tqdm.
            progress_kwargs: Keyword arguments passed to tqdm.
            kwargs: Additional keyword arguments for the
                emcee.EnsembleSampler.sample method.
        """
        p0 = self._get_random_walk(n_walkers=n_walkers)
        log_likelihood = self.get_log_likelihood_function()

        # TODO: sampling with correlated priors
        sampler = EnsembleSampler(
            n_walkers, self.n_dim, log_likelihood
        )
        state = sampler.run_mcmc(
            initial_state=p0, nsteps=n_burn_in_steps,
        )
        sampler.reset()
        sampler.run_mcmc(
            initial_state=state,
            nsteps=n_steps,
            progress=progress,
            progress_kwargs=progress_kwargs,
            **kwargs
        )

        self.sampler = sampler


    def get_chain(self, flat=True):
        """Get the chain of samples from the MCMC sampler."""
        if self.sampler is None:
            raise ValueError("MCMC sampler has not been run yet.")
        else:
            return self.sampler.get_chain(flat=flat)

    def save_chain(self, flat=True, fname="mcmc_chain.npy"):
        """Save the chain of samples from the MCMC sampler."""
        if self.sampler is None:
            raise ValueError("MCMC sampler has not been run yet.")
        else:
            np.save(fname, self.sampler.get_chain(flat=flat))

    def save_log_prob(self, flat=True, fname="mcmc_log_prob.npy"):
        """Save the log probabilities of the samples from the MCMC sampler."""
        if self.sampler is None:
            raise ValueError("MCMC sampler has not been run yet.")
        else:
            np.save(fname, self.sampler.get_log_prob(flat=flat))


def Hartlap_factor(n_simulations: int, dv_length: int) -> float:
    """Calculate the Hartlap factor for covariance matrix correction."""
    return (n_simulations - 1) / (n_simulations - dv_length - 2)


def Sellentin_Heavens_factor(
        n_simulations: int,
        dv_length: int,
        n_parameters: int,
):
    """Calculate the Sellentin-Heavens factor for covariance matrix correction."""
    return (n_simulations - 1) / (n_simulations - dv_length + n_parameters - 1)


def compute_quantiles(
        samples: np.ndarray,
        quantiles: tuple[float]=(0.16, 0.5, 0.84),
) -> list[float]:
    """Compute sample quantiles.

    Args:
        samples: Samples from the posterior distribution. Can be a 1D array of
            samples for a single parameter, or a 2D array of shape
            (n_samples, n_params).
        quantiles:
    """
    if np.ndim(samples) == 1:
        samples = samples[:, np.newaxis] # convert to shape (n_samples, 1)
    quantiles = np.asarray(quantiles)

    qvalues = []

    if not all((0. < quantile < 1.) for quantile in quantiles):
        raise ValueError("Quantiles must be between 0 and 1")

    for param_samples in samples.T:
        qvalues_i = np.percentile(param_samples, list(100 * quantiles))
        qvalues.append(qvalues_i.tolist())

    if len(qvalues) == 1:
        return qvalues[0]
    else:
        return qvalues


def estimate_MAP(
        samples: np.ndarray,
        method: str="kde",
        log_prob: np.ndarray | None=None,
        kde_kwargs: dict | None=None,
) -> list[float]:
    """Estimate the maximum a posteriori (MAP) estimate from the samples according
    to the log probabilities.
    """
    n_samples, n_params = samples.shape
    match method:
        case "map_sample":
            if log_prob is None:
                raise ValueError("For sample_map method, log_prob must be provided.")
            assert len(samples) == len(log_prob)
            return samples[np.argmax(log_prob)].tolist()

        case "knn":
            from sklearn.neighbors import NearestNeighbors
            k = max(10, int(np.sqrt(n_samples)))
            nbrs = NearestNeighbors(n_neighbors=k + 1).fit(samples)
            dist, _ = nbrs.kneighbors(samples)
            dist_safe = dist[:, 1:] + 1.e-10
            dens = np.mean(1. / dist_safe, axis=1)
            return samples[np.argmax(dens)].tolist()

        case "meanshift":
            raise NotImplementedError

        case "kde":
            from seaborn._statistics import KDE
            MAPs = []
            for param_samples in samples.T:
                kde = KDE(**kde_kwargs)
                dens, support = kde(param_samples)
                MAPs.append(float(support[np.argmax(dens)]))
            return MAPs

        case _:
            raise ValueError(f"Unknown method for MAP estimation.")

