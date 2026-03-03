import torch
import warnings
import numpy as np
import pickle as pk

from scatterlens.mcmc import Hartlap_factor, Sellentin_Heavens_factor

warnings.filterwarnings("ignore", category=RuntimeWarning)

def greedy(
        jacobian: np.ndarray | torch.Tensor,
        cov: np.ndarray | torch.Tensor,
        iteration_mode: str="tomography",
        n_iterations: int | None=None,
        zbin_combos: list[tuple[int, ...]] | None=None,
        Hartlap_correction: bool=False,
        Sellentin_Heavens_correction: bool=False,
        n_simulations: int | None=None,
        n_parameters: int | None=None,
        savepath: str=None
):
    """Greedy algorithm for data compression.

    Args:
        jacobian:
        cov:
        iteration_mode:
        n_iterations:
        zbin_combos:
        Hartlap_correction:
        Sellentin_Heavens_correction:
        n_simulations:
        n_parameters:
        savepath:

    Returns:
        If iteration_mode is "tomography", returns a list of indices of the
        data vector elements to keep, and an ordered list of zbin_combos
        corresponding to the kept data vector elements, and a list of
        FoM values.
    """
    assert iteration_mode in ["tomography", "dv_elements"]
    assert not (Hartlap_correction and Sellentin_Heavens_correction)

    if isinstance(cov, torch.Tensor):
        cov = cov.numpy()

    if iteration_mode == "tomography":
        assert zbin_combos is not None
        n_iterations = n_iterations or len(zbin_combos)
        dv_length_per_combo = jacobian.shape[0] // len(zbin_combos)
        FoM_values = []
        dv_indices = []
        zbin_combos_ordered = []

        for iteration in range(n_iterations):
            FoM_ = []
            for zbin_combo in zbin_combos:
                if zbin_combo in zbin_combos_ordered:
                    FoM_.append(0.)
                else:
                    indices = dv_indices + [*range(
                        dv_length_per_combo * zbin_combos.index(zbin_combo),
                        dv_length_per_combo * (zbin_combos.index(zbin_combo) + 1),
                    )]
                    jacobian_slice = jacobian[indices, :]
                    cov_slice = cov[np.ix_(indices, indices)]
                    if Hartlap_correction:
                        assert n_simulations is not None
                        cov_slice *= Hartlap_factor(n_simulations, len(indices))

                    if Sellentin_Heavens_correction:
                        assert n_simulations is not None
                        assert n_parameters is not None
                        cov_slice *= Sellentin_Heavens_factor(
                            n_simulations, len(indices), n_parameters)

                    try:
                        np.linalg.inv(cov_slice)
                    except np.linalg.LinAlgError:
                        print(
                            f"Covariance non-invertible.",
                            f"Stopping iteration {iteration} for zbin_combo {zbin_combo}."
                        )
                        return zbin_combos_ordered, dv_indices, np.array(FoM_values)

                    fisher = jacobian_slice.T @ np.linalg.inv(cov_slice) @ jacobian_slice

                    if (
                            np.isnan(np.linalg.det(fisher)) or
                            np.linalg.det(fisher) < 0 or
                            np.linalg.det(fisher) > 1.e6
                    ):
                        return zbin_combos_ordered, dv_indices, np.array(FoM_values)
                    else:
                        FoM = np.sqrt(np.linalg.det(fisher))
                        FoM_.append(FoM)

            FoM_ = np.array(FoM_)
            optm_zbin_combo = zbin_combos[np.argmax(FoM_)]
            dv_indices += [*range(
                dv_length_per_combo * zbin_combos.index(optm_zbin_combo),
                dv_length_per_combo * (zbin_combos.index(optm_zbin_combo) + 1),
            )]
            zbin_combos_ordered.append(optm_zbin_combo)
            FoM_values.append(FoM_[np.argmax(FoM_)])

        FoM_values = np.array(FoM_values)

        if savepath:
            with open(savepath, "wb") as f:
                pk.dump(obj={
                    "zbin_combos_ordered": zbin_combos_ordered,
                    "dv_indices": dv_indices,
                    "FoM_values": FoM_values,
                }, file=f)

        return zbin_combos_ordered, dv_indices, FoM_values


    if iteration_mode == "dv_elements":
        raise NotImplementedError