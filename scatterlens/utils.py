import multiprocessing
import numpy as np
from pymaster.utils import mask_apodization_flat as _mask_apodization_flat
from tqdm import tqdm

def mp_wapper_calc_scoef(stlib, *args):
    """Wapper function of scatter coefficients calculation for the use of
    multiprocessing."""
    assert hasattr(stlib, "calc_sim_scoef")
    assert callable(stlib.calc_sim_scoef)
    return stlib.calc_sim_scoef(*args)


def run_mp_scattering(
        cross_redshift_bins: list[tuple[int, int]],
        regions: list[str],
        cosmolstlib=None,
        cosmols: list | None=None,
        covstlib=None,
):
    """Function to run the scatter calculation using multiprocessing."""
    assert (cosmolstlib and cosmols) or covstlib

    args_list = []
    if cosmolstlib:
        cosmo_args_list = [
            (cosmolstlib, cosmol, zbin1, zbin2, region)
            for cosmol in cosmols
            for (zbin1, zbin2) in cross_redshift_bins
            for region in regions
        ]
        args_list += cosmo_args_list

    if covstlib:
        cov_args_list = [
            (covstlib, zbin1, zbin2, region)
            for (zbin1, zbin2) in cross_redshift_bins
            for region in regions
        ]
        args_list += cov_args_list

    multiprocessing.set_start_method("spawn")
    pool = multiprocessing.Pool(processes=5)

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