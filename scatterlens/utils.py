import multiprocessing
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

    with tqdm(total=len(args_list), **tqdm_style) as pbar:
        for async_result in async_results:
            async_result.get()
            pbar.update()


tqdm_style = {
    "ncols": 80,
    "ascii": " =",
    "smoothing": 0.,
}
