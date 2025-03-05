import os
import numpy as np
import pandas as pd
from numpy.typing import NDArray

_data_path = "/data1/jliu/scattering-transform-kids/data/KiDS-1000/MassMaps4JHD/"

_zbin_to_ShapeNoise = {
    1: 0.270,
    2: 0.258,
    3: 0.273,
    4: 0.254,
    5: 0.270,
}

_zbin_to_zrange = {
    1: "0.1-0.3",
    2: "0.3-0.5",
    3: "0.5-0.7",
    4: "0.7-0.9",
    5: "0.9-1.2",
}

def _has_cosmol(cosmol: int) -> bool:
    # In SLICE simulation set has 25 wCDM cosmologies and one Lambda-CDM cosmology.
    # -1 points to fiducial cosmology
    return -1 <= cosmol <= 24

def _has_region(region: int) -> bool:
    return 1 <= region <= 18

def _has_zbin(zbin: int) -> bool:
    return isinstance(zbin, int) and 1 <= zbin <= 5

class CosmoSLICE:
    resol = 2.344
    resol_unit = "arcmin"
    simspath = "MRres140.64arcs_100Sqdeg_SN{:g}_Mosaic_KiDS1000GpAM_zKiDS1000_{:s}_Cosmol{:d}"
    simspath_fid = "MRres140.64arcs_100Sqdeg_SN{:g}_Mosaic_KiDS1000GpAM_zKiDS1000_{:s}_Cosmolfid"
    mapfname = r"SN{:g}_Mosaic.KiDS1000GpAM.LOS{}R{:d}.SS2.816.Ekappa.npy"

    @staticmethod
    def get_sim_massmap(
            cosmol: int,
            zbin1: int,
            zbin2: int,
            LOS: int,
            region: int,
    ) -> NDArray:
        assert (
            _has_cosmol(cosmol)
            and _has_region(region)
            and _has_zbin(zbin1)
            and _has_zbin(zbin2)
            and CosmoSLICE._has_LOS(LOS)
        ), "Validation failed for one or more inputs."

        shapenoise = _zbin_to_ShapeNoise[zbin1]
        zbcut = f"ZBcut{_zbin_to_zrange[zbin1]}"
        zbcut += f"_X_ZBcut{_zbin_to_zrange[zbin2]}" if zbin2 != zbin1 else ""

        if cosmol==-1:
            simspath = CosmoSLICE.simspath_fid.format(shapenoise, zbcut)
        else:
            simspath = CosmoSLICE.simspath.format(shapenoise, zbcut, cosmol)

        massmap = np.load(
            os.path.join(
                _data_path, simspath,
                CosmoSLICE.mapfname.format(shapenoise, LOS, region),
            )
        )
        return massmap

    @staticmethod
    def get_fid_massmap(
            zbin1: int,
            zbin2: int,
            LOS: int,
            region: int,
    ) -> NDArray:
        massmap = CosmoSLICE.get_sim_massmap(cosmol=-1, zbin1=zbin1,
            zbin2=zbin2, LOS=LOS, region=region)
        return massmap

    @staticmethod
    def _has_LOS(LOS: int) -> bool:
        return 1 <= LOS <= 50



class SLICE:
    @staticmethod
    def get_sim_kappa(self, region):
        pass

    @staticmethod
    def _has_LOS(LOS: int) -> bool:
        pass


def _read_cosmologies_info():
    fname = os.path.join(os.path.dirname(__file__), "data", "kids1000_cosmol.csv")
    df = pd.read_csv(
        fname, delimiter='\t', skiprows=1,
        names=['id', 'Omega_m', 'S_8', 'h', 'w_0', 'sigma_8', 'Omega_cdm'],
    )
    return df

cosmologies = _read_cosmologies_info()
