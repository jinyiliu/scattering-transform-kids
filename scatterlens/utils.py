
def mp_wapper_calc_scoef(stlib, *args):
    """Wapper function of scatter coefficients calculation for the use of
    multiprocessing."""
    assert hasattr(stlib, "calc_sim_scoef")
    assert callable(stlib.calc_sim_scoef)
    return stlib.calc_sim_scoef(*args)

tqdm_style = {
    "ncols": 80,
    "ascii": " =",
    "smoothing": 0.,
}
