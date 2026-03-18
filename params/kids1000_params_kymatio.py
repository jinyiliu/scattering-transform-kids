import torch

from scatterlens.kids1000_sims import *
from scatterlens.library import CosmolStLibrary, FilterLibrary, MaskLibrary, CovStLibrary, IAStLibrary

from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor


# Wavelet parameters
J = 3
L = 4
Q = 3. / 5. * np.pi
sigma_0 = 0.8
dilation_factor = 2.0

# Scattering scales to save
j_start = 2
j_end = J - 1

# Mask and padding parameters
apotype = "C2"
aposcale = 5.0 # arcmin
padding = 40 # pixel
dtype = torch.float64


_libdir = f"/data2/jliu/stkids_data/KiDS-1000_StLib_{aposcale:.0f}amin{apotype}apo_{padding}pixZeropad/"
cosmol_libdir = _libdir + "CosmoSlicsSt"
filter_libdir = _libdir + "Filters"
mask_libdir = _libdir + "Masks"
cov_libdir = _libdir + "SlicsSt"
ia_libdir = _libdir + "IASt"

fig_savedir = _libdir + "Figures/"
inference_savedir = _libdir + "Inference"


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

IAStLib = IAStLibrary(
    libdir=ia_libdir,
    sims=IAMocks,
    padding=0,
    J=J,
    L=L,
    dtype=dtype,
    Q=Q,
    sigma_0=sigma_0,
    dilation_factor=dilation_factor,
)


# Emulator parameters
regressor_type = GaussianProcessRegressor
gpr_kernel = 0.1**2 * Matern()
target_scaler = StandardScaler()
input_scaler = MinMaxScaler()

# MCMC parameters
likelihood_type = "Sellentin_Heavens"
n_walkers = 100
n_steps = 1000

param_priors = {
    "Omega_m": ["flat", CosmoSLICS.param_ranges["Omega_m"]],
    "S_8": ["flat", CosmoSLICS.param_ranges["S_8"]],
    "h": ["flat", CosmoSLICS.param_ranges["h"]],
    "w_0": ["flat", CosmoSLICS.param_ranges["w_0"]],
}