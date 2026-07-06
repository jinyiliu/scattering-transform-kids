import torch
import warnings
import numpy as np
import pickle as pk
from tqdm import tqdm

from scatterlens.mcmc import Hartlap_factor, Sellentin_Heavens_factor
from scatterlens.utils import _tqdm_style

warnings.filterwarnings("ignore", category=RuntimeWarning)

def _apply_corrections(
        cov_slice: np.ndarray,
        n_elem: int,
        Hartlap_correction: bool,
        Sellentin_Heavens_correction: bool,
        n_simulations: int,
        n_params: int,
):
    if Hartlap_correction:
        assert n_simulations is not None
        cov_slice = cov_slice * Hartlap_factor(
            n_simulations,
            n_elem,
        )
    if Sellentin_Heavens_correction:
        assert n_simulations is not None
        cov_slice = cov_slice * Sellentin_Heavens_factor(
            n_simulations,
            n_elem,
            n_params,
        )
    return cov_slice


def _compute_fom(
        jac_slice: np.ndarray,
        cov_slice: np.ndarray,
        n_params: int,
        force_det,
):
    fisher = jac_slice.T @ np.linalg.inv(cov_slice) @ jac_slice
    if not force_det and jac_slice.shape[0] < n_params:
        return np.trace(fisher)
    det = np.linalg.det(fisher)
    if np.isnan(det) or det < 0 or det > 1.e6:
        return None
    return np.sqrt(det)


def greedy(
        jacobian: np.ndarray | torch.Tensor,
        cov: np.ndarray | torch.Tensor,
        iteration_mode: str="tomography",
        n_iterations: int | None=None,
        zbin_combos: list[tuple[int, ...]] | None=None,
        Hartlap_correction: bool=False,
        Sellentin_Heavens_correction: bool=False,
        n_simulations: int | None=None,
        J: int | None=None,
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
        J:
        savepath:

    Returns:
        If iteration_mode is "tomography": (zbin_combos_ordered, dv_indices, FoM_values).
        If iteration_mode is "dv_elements": (dv_descriptors, dv_indices, FoM_values).
    """
    assert iteration_mode in ["tomography", "dv_elements"]
    assert not (Hartlap_correction and Sellentin_Heavens_correction)

    if isinstance(cov, torch.Tensor):
        cov = cov.numpy()

    n_params = jacobian.shape[1]
    dv_length = jacobian.shape[0]
    dv_length_per_combo = dv_length // len(zbin_combos)
    FoM_values = []
    dv_indices = []

    if iteration_mode == "tomography":
        assert zbin_combos is not None
        n_iterations = n_iterations or len(zbin_combos)
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
                    cov_slice = _apply_corrections(
                        cov_slice,
                        len(indices),
                        Hartlap_correction,
                        Sellentin_Heavens_correction,
                        n_simulations,
                        n_params,
                    )
                    try:
                        fom = _compute_fom(
                            jacobian_slice, cov_slice, n_params, force_det=True,
                        )
                    except np.linalg.LinAlgError:
                        print(
                            f"Covariance non-invertible.",
                            f"Stopping iteration {iteration} for zbin_combo {zbin_combo}."
                        )
                        return zbin_combos_ordered, dv_indices, np.array(FoM_values)
                    if fom is None:
                        return zbin_combos_ordered, dv_indices, np.array(FoM_values)
                    else:
                        FoM_.append(fom)

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
        if J is None:
            raise ValueError("J must be provided")
        n_iterations = n_iterations or dv_length

        # Build flat descriptor mapping
        _descriptors = []
        S2_pairs = [(j1, j2) for j1 in range(J) for j2 in range(j1 + 1, J)]
        for zbin_combo in zbin_combos:
            for n in range(dv_length_per_combo):
                if n < J:
                    _descriptors.append([zbin_combo, n])
                else:
                    j1, j2 = S2_pairs[n - J]
                    _descriptors.append([zbin_combo, j1, j2])


        for iteration in tqdm(range(n_iterations), desc="dv_elements", **_tqdm_style):
            FoM_ = []
            for idx in range(dv_length):
                if idx in dv_indices:
                    FoM_.append(0.)
                else:
                    indices = dv_indices + [idx]
                    jacobian_slice = jacobian[indices, :]
                    cov_slice = cov[np.ix_(indices, indices)]
                    cov_slice = _apply_corrections(
                        cov_slice, len(indices),
                        Hartlap_correction, Sellentin_Heavens_correction,
                        n_simulations, n_params,
                    )
                    try:
                        fom = _compute_fom(
                            jacobian_slice, cov_slice, n_params, force_det=False,
                        )
                    except np.linalg.LinAlgError:
                        print(
                            f"Covariance non-invertible.",
                            f"Stopping iteration {iteration} for index {idx}.",
                        )
                        FoM_values = np.array(FoM_values)
                        dv_descriptors = [_descriptors[i] for i in dv_indices]
                        if savepath:
                            with open(savepath, "wb") as f:
                                pk.dump(obj={
                                    "dv_indices": dv_indices,
                                    "FoM_values": FoM_values,
                                    "dv_descriptors": dv_descriptors,
                                }, file=f)
                        return dv_descriptors, dv_indices, FoM_values
                    if fom is None:
                        FoM_values = np.array(FoM_values)
                        dv_descriptors = [_descriptors[i] for i in dv_indices]
                        if savepath:
                            with open(savepath, "wb") as f:
                                pk.dump(obj={
                                    "dv_indices": dv_indices,
                                    "FoM_values": FoM_values,
                                    "dv_descriptors": dv_descriptors,
                                }, file=f)
                        return dv_descriptors, dv_indices, FoM_values
                    else:
                        FoM_.append(fom)

            FoM_ = np.array(FoM_)
            optm_idx = int(np.argmax(FoM_))
            dv_indices.append(optm_idx)

            if len(dv_indices) < n_params:
                FoM_values.append(0.)
            else:
                FoM_values.append(FoM_[optm_idx])

        FoM_values = np.array(FoM_values)
        dv_descriptors = [_descriptors[i] for i in dv_indices]

        if savepath:
            with open(savepath, "wb") as f:
                pk.dump(obj={
                    "dv_descriptors": dv_descriptors,
                    "dv_indices": dv_indices,
                    "FoM_values": FoM_values,
                }, file=f)

        return dv_descriptors, dv_indices, FoM_values