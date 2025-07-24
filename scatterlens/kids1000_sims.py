import os
import numpy as np
import pandas as pd
from numpy.typing import NDArray

_data_path = "/data1/jliu/scattering-transform-kids/data/KiDS-1000/MassMaps4JHD/"

class KiDS1000:
    resol = 2.344
    resol_unit = "arcmin"
    region_indices = list(range(1, 19))
    zbin_indices = list(range(0, 6))
    zbin_to_ShapeNoise = {
        0: 0.265,
        1: 0.270,
        2: 0.258,
        3: 0.273,
        4: 0.254,
        5: 0.270,
    }
    zbin_to_zrange = {
        0: "0.1-1.2",
        1: "0.1-0.3",
        2: "0.3-0.5",
        3: "0.5-0.7",
        4: "0.7-0.9",
        5: "0.9-1.2",
    }
    region_MN = {
        1: (127, 115),
        2: (127, 221),
        3: (152, 255),
        4: (152, 255),
        5: (152, 255),
        6: (152, 255),
        7: (152, 255),
        8: (152, 255),
        9: (177, 255),
        10: (103, 255),
        11: (224, 253),
        12: (205, 252),
        13: (205, 251),
        14: (205, 250),
        15: (180, 250),
        16: (180, 250),
        17: (223, 253),
        18: (224, 159),
    }

    cross_zbins: list[tuple[int, int]] = []
    for zbin1 in zbin_indices:
        for zbin2 in zbin_indices:
            if zbin1 == 0:
                if zbin2 == 0:
                    cross_zbins.append((zbin1, zbin2))
                else:
                    continue
            else:
                if zbin2 >= zbin1:
                    cross_zbins.append((zbin1, zbin2))

    @staticmethod
    def has_region(region: int) -> bool:
        return 1 <= region <= 18

    @staticmethod
    def has_zbin(zbin: int) -> bool:
        return isinstance(zbin, int) and 0 <= zbin <= 5


class CosmoSLICE(KiDS1000):
    simsname = "KiDS1000_CosmoSLICE"
    simspath = "MRres140.64arcs_100Sqdeg_SN{:g}_Mosaic_KiDS1000GpAM_zKiDS1000_{:s}_Cosmol{:d}"
    simspath_fid = "MRres140.64arcs_100Sqdeg_SN{:g}_Mosaic_KiDS1000GpAM_zKiDS1000_{:s}_Cosmolfid"
    mapfname = r"SN{:g}_Mosaic.KiDS1000GpAM.LOS{}R{:d}.SS2.816.Ekappa.npy"
    cosmol_indices = list(range(-1, 25))
    LOS_indices = list(range(1, 51))
    param_ranges = {
        "Omega_m": [0.1, 0.55],
        "S_8": [0.6, 0.9],
        "h": [0.6, 0.9],
        "w_0": [-2.0, 0.5],
    }

    @staticmethod
    def get_sim_massmap(
            cosmol: int,
            zbin1: int,
            zbin2: int,
            region: int,
            LOS: int,
    ) -> NDArray:
        assert (
            CosmoSLICE.has_cosmol(cosmol)
            and KiDS1000.has_region(region)
            and KiDS1000.has_zbin(zbin1)
            and KiDS1000.has_zbin(zbin2)
            and CosmoSLICE.has_LOS(LOS)
        ), "Validation failed for one or more inputs."

        shapenoise = KiDS1000.zbin_to_ShapeNoise[zbin1]
        zbcut = f"ZBcut{KiDS1000.zbin_to_zrange[zbin1]}"
        zbcut += f"_X_ZBcut{KiDS1000.zbin_to_zrange[zbin2]}" if zbin2 != zbin1 else ""

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
            region: int,
            LOS: int,
    ) -> NDArray:
        massmap = CosmoSLICE.get_sim_massmap(cosmol=-1, zbin1=zbin1,
            zbin2=zbin2, LOS=LOS, region=region)
        return massmap

    @staticmethod
    def has_LOS(LOS: int) -> bool:
        return 1 <= LOS <= 50

    @staticmethod
    def has_cosmol(cosmol: int) -> bool:
        # SLICE simulation set has 25 wCDM cosmologies and one Lambda-CDM cosmology.
        # -1 points to fiducial cosmology
        return -1 <= cosmol <= 24

    @staticmethod
    def cosmology_info(cosmol: int | str | None=None):
        if cosmol:
            if isinstance(cosmol, str):
                assert cosmol == "fid"
                return _cosmologies.loc[25]

            if isinstance(cosmol, int):
                assert CosmoSLICE.has_cosmol(cosmol)
                if cosmol == -1:
                    return _cosmologies.loc[25]
                else:
                    return _cosmologies.loc[cosmol]
        else:
            return _cosmologies.loc[:25]


class SLICE(KiDS1000):
    simsname = "KiDS1000_SLICE"
    simspath = "MRres140.64arcs_100Sqdeg_SN{:g}_Mosaic_KiDS1000GpAM_zKiDS1000_{:s}"
    mapfname = r"SN{:g}_Mosaic.KiDS1000GpAM.LOS{}R{:d}.SS2.816.Ekappa.npy"
    LOS_indices = list(range(74, 293))
    LOS_indices.pop(198 - 74) # LOS198 & LOS199 are corrupted
    LOS_indices.pop(198 - 74)

    @staticmethod
    def get_sim_massmap(zbin1: int, zbin2: int, region: int, LOS: int) -> NDArray:
        assert (
            KiDS1000.has_region(region)
            and KiDS1000.has_zbin(zbin1)
            and KiDS1000.has_zbin(zbin2)
            and SLICE.has_LOS(LOS)
        ), "Validation failed for one or more inputs."
        shapenoise = KiDS1000.zbin_to_ShapeNoise[zbin1]
        zbcut = f"ZBcut{KiDS1000.zbin_to_zrange[zbin1]}"
        zbcut += f"_X_ZBcut{KiDS1000.zbin_to_zrange[zbin2]}" if zbin2 != zbin1 else ""

        simspath = SLICE.simspath.format(shapenoise, zbcut)

        massmap = np.load(
            os.path.join(
                _data_path, simspath,
                SLICE.mapfname.format(shapenoise, LOS, region),
            )
        )
        return massmap

    @staticmethod
    def has_LOS(LOS: int) -> bool:
        return 74 <= LOS <= 292 and not LOS in (198, 199)

    @staticmethod
    def cosmology_info():
        return _cosmologies.loc[26]


def _read_cosmologies_info() -> pd.DataFrame:
    fname = os.path.join(os.path.dirname(__file__), "data", "kids1000_cosmol.csv")
    df = pd.read_csv(
        fname, delimiter='\t', skiprows=1,
        names=['id', 'Omega_m', 'S_8', 'h', 'w_0', 'sigma_8', 'Omega_cdm'],
    )
    return df

_cosmologies = _read_cosmologies_info()
