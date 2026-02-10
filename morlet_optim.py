"""
Optimise Morlet wavelet parameters for KiDS-1000 mock data.
"""
import torch
import warnings
import numpy as np
from scipy.optimize import fsolve
from derivkit.forecast_kit import ForecastKit

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.exceptions import ConvergenceWarning

from scatterlens.wavelets import Morlet2D, multiple2freq
from scatterlens.library import FilterLibrary, MaskLibrary, CosmolStLibrary, CovStLibrary
from scatterlens.emulator import PerFeatureEmulator
from scatterlens.kids1000_sims import CosmoSLICS, SLICS, KiDS1000
from scatterlens.utils import run_mp_scattering, FoM_SLICS_Fisher_Omega_m_and_S_8
from scatterlens.data.morlet_optim_x0 import x0_

warnings.filterwarnings("ignore", category=ConvergenceWarning)

zbin_combos = [(3,), (4,), (5,)]
KiDS1000.zbin_combos = zbin_combos

lmin = 100
lmax = 1600

def solve_boundary_conditions(J: int, Q: float, x0: list[float]=None):
    """Solve for the dilation factor and centre frequency of the mother wavelet
    given the boundary conditions.

    Args:
        J: Number of scales.
        Q: Quality factor of the Morlet wavelet.
        x0: Initial guess for [dilation_factor, xi_0].

    Returns:
        The dilation factor and the centre frequency of the mother wavelet.
    """
    if x0 is None:
        x0 = np.array(x0_[str(J)][f"{Q:.2f}"])
    else:
        x0 = np.array(x0)

    psi_max = 1 - np.exp(-Q**2)

    def boundary_conditions(x):
        dilation_factor, xi_0 = x
        psi_lmax = Morlet2D.get_profile(
            j=0,
            freq=multiple2freq(lmax, KiDS1000.pixel_length),
            Q=Q,
            sigma_0=Q / xi_0,
            dilation_factor=dilation_factor,
        )
        psi_lmin = Morlet2D.get_profile(
            j=J - 1,
            freq=multiple2freq(lmin, KiDS1000.pixel_length),
            Q=Q,
            sigma_0=Q / xi_0,
            dilation_factor=dilation_factor,
        )
        condition1 = psi_lmax - psi_max / 2
        condition2 = psi_lmin - psi_max / 2
        return [condition1, condition2]

    dilation_factor, xi_0 = fsolve(boundary_conditions, x0=x0)
    return dilation_factor, xi_0


def compute_FoM_given_J_and_Q(J: int, Q: float, run_scattering: bool=True):
    """Compute the figure of merit (FoM) for given J and Q."""
    L = 4
    dilation_factor, xi_0 = solve_boundary_conditions(J, Q)
    sigma_0 = Q / xi_0
    print(f"J={J:d}, Q={Q:.3f}, dilation_factor={dilation_factor:.3f}, xi_0={xi_0:.3f}, sigma_0={sigma_0:.3f}")

    apotype = "C2"
    aposcale = 5.0 # arcmin
    padding = 40 # pixel
    dtype = torch.float64

    _libdir = f"/data2/jliu/stkids_data/MorletOptim/J{J}_Q{Q:.2f}/"
    cosmol_libdir = _libdir + "CosmoSlicsSt"
    filter_libdir = _libdir + "Filters"
    mask_libdir = _libdir + "Masks"
    cov_libdir = _libdir + "SlicsSt"

    FilterLib = FilterLibrary(
        libdir=filter_libdir,
        region_MN=CosmoSLICS.region_MN,
        padding=padding,
        J=J,
        L=L,
        dtype=dtype,
        Q=Q,
        sigma_0=sigma_0,
        dilation_factor=dilation_factor,
    )

    MaskLib = MaskLibrary(
        libdir=mask_libdir,
        apotype=apotype,
        aposcale=aposcale,
        sims=CosmoSLICS,
        dtype=dtype,
    )

    CosmolStLib = CosmolStLibrary(
        libdir=cosmol_libdir,
        filterlib=FilterLib,
        masklib=MaskLib,
        sims=CosmoSLICS,
        padding=padding,
        J=J,
        L=L,
        dtype=dtype,
    )

    CovStLib = CovStLibrary(
        libdir=cov_libdir,
        filterlib=FilterLib,
        masklib=MaskLib,
        sims=SLICS,
        padding=padding,
        J=J,
        L=L,
        dtype=dtype,
    )

    regressor_type = GaussianProcessRegressor
    gpr_kernel = 0.1 ** 2 * Matern()
    target_scaler = StandardScaler()
    input_scaler = MinMaxScaler()

    if run_scattering:
        run_mp_scattering(
            cosmolstlib=CosmolStLib,
            processes=5,
        )

        run_mp_scattering(
            covstlib=CovStLib,
            processes=2,
        )

    cov = CovStLib.get_cov(
        zbin_combos=zbin_combos,
        decorrelated_S2=False,
        Hartlap_correction=True,
    )

    ml_training_set = CosmolStLib.get_ml_training_set(
        region_weights="auto",
        drop_S0=True,
        decorrelated_S2=False,
    )
    emu = PerFeatureEmulator(
        training_set=ml_training_set,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        regressor_type=regressor_type,
        kernel=gpr_kernel,
        n_restarts_optimizer=50,
    )
    emu.fit()

    return FoM_SLICS_Fisher_Omega_m_and_S_8(emulator=emu, cov=cov)


if __name__=="__main__":
    J_values = [2, 3, 4, 5, 6, 7]
    Q_values = [fac * np.pi / 10 for fac in range(4, 19)]

    fom_values = np.empty((len(J_values), len(Q_values)))

    for i, J in enumerate(J_values):
        for j, Q in enumerate(Q_values):
            fom = compute_FoM_given_J_and_Q(
                J=J, Q=Q, run_scattering=False)
            fom_values[i, j] = fom

    np.save(
        file="/data1/jliu/scattering-transform-kids/data/MorletOptim/FoM_values.npy",
        arr=fom_values,
    )