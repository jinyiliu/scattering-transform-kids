import torch
import multiprocessing
import numpy as np
from pymaster.utils import mask_apodization_flat as _mask_apodization_flat
from tqdm import tqdm
from itertools import combinations

def mp_wapper_calc_scoef(stlib, *args):
    """Wapper function of scatter coefficients calculation for the use of
    multiprocessing."""
    assert hasattr(stlib, "calc_sim_scoef")
    assert callable(stlib.calc_sim_scoef)
    return stlib.calc_sim_scoef(*args)


def run_mp_scattering(
        cosmolstlib=None,
        covstlib=None,
        processes: int=1,
):
    """Function to run the scatter calculation using multiprocessing.

    Args:
        cosmolstlib: Instance of scatterlens.library.CosmolstLibrary.
        covstlib: Instance of scatterlens.library.CovstLibrary.
        processes: Number of processes to use.
    """
    if not (cosmolstlib or covstlib):
        raise ValueError("Must specify at least one cosmolstlib or covstlib")

    args_list = []
    if cosmolstlib:
        cosmo_args_list = [
            (cosmolstlib, cosmol, zbin_combo, region)
            for cosmol in cosmolstlib.sims.cosmol_indices
            for zbin_combo in cosmolstlib.sims.zbin_combos
            for region in cosmolstlib.sims.region_indices
        ]
        args_list += cosmo_args_list

    if covstlib:
        cov_args_list = [
            (covstlib, zbin_combo, region)
            for zbin_combo in covstlib.sims.zbin_combos
            for region in covstlib.sims.region_indices
        ]
        args_list += cov_args_list

    multiprocessing.set_start_method("spawn")
    pool = multiprocessing.Pool(processes=processes)

    async_results = [
        pool.apply_async(mp_wapper_calc_scoef, args) for args in args_list
    ]

    with tqdm(total=len(args_list), **_tqdm_style) as pbar:
        for async_result in async_results:
            async_result.get()
            pbar.update()


_tqdm_style = {
    "ncols": 80,
    "ascii": " =",
    "smoothing": 0.,
}


def create_zbin_combos(zbin_indices: list[int], r_max: int=np.inf) -> list[tuple[int, ...]]:
    """Generate all redshift bin comibinations from single to multiple bins.

    Args:
        zbin_indices: List of redshift bin indices.
        r_max: Maximum number of redshift bins in a combination.

    Example:
        If the input is [1, 2, 3], the output will be:
            [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
        If the input is [1, 2, 3] and r_max=2, the output will be:
            [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3)]
    """
    zbin_combos = []

    for r in range(1, min(len(zbin_indices), r_max) + 1):
        combos = combinations(zbin_indices, r)
        zbin_combos.extend(combos)

    return zbin_combos


def cov2corr(cov: torch.Tensor) -> torch.Tensor:
    """Convert a covariance matrix to a correlation matrix with diagonal
    elements equal to one."""
    diag = torch.diagonal(cov, dim1=-2, dim2=-1)
    std_devs = torch.sqrt(diag)
    denom = std_devs.unsqueeze(-1) * std_devs.unsqueeze(-2)
    corr = cov / denom
    return corr


def mask_apodization(
        mask: np.ndarray,
        xresol: float,
        yresol: float,
        aposcale: float,
        apotype: str="C1",
) -> np.ndarray:
    """Apodize a mask with a given apodization scale.

    Args:
        mask: A binary mask.
        xresol: Resolution of the mask in x-axis in arcminutes.
        yresol: Resolution of the mask in y-axis in arcminutes.
        aposcale: Apodization scale in arcminutes.
        apotype: Apodization type in ["C1", "C2", "Smooth"].

    Returns:
        Apodized mask.

    Notes:
        This is a wrapper for mask_apodization_flat function in pymaster.utils.
    """
    lx = xresol * mask.shape[0] / 180 / 60
    ly = yresol * mask.shape[1] / 180 / 60
    aposcale_deg = aposcale / 60
    return _mask_apodization_flat(mask, lx, ly, aposcale_deg, apotype)