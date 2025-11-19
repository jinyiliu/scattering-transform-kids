import os
import numpy as np
from numpy.typing import NDArray

_data_path = "/data1/jliu/scattering-transform-kids/data/KiDS-Legacy/MassMaps4JHD/"

class KiDSLegacy:
    smoothing_scale = 6.6
    smoothing_scale_unit = "arcmin"
    pixel_length = 2.344
    pixel_length_unit = "arcmin"
    region_indices = list(range(1, 19))
    zbin_indices = list(range(1, 7))
    region_MN = {
        1: (127, 115),
        2: (127, 222),
        3: (203, 256),
        4: (203, 256),
        5: (203, 256),
        6: (203, 256),
        7: (203, 256),
        8: (203, 256),
        9: (203, 256),
        10: (103, 256),
        11: (224, 254),
        12: (224, 254),
        13: (224, 254),
        14: (224, 254),
        15: (224, 267),
        16: (224, 254),
        17: (224, 254),
        18: (224, 159),
    }

    zbin_combos = [(5,)]

    _zbin_combo_to_zrange = {
        (5,): "0.9-1.14",
    }

    @staticmethod
    def has_region(region: int) -> bool:
        return 1 <= region <= 18

    @staticmethod
    def has_zbin_combo(zbin_combo: tuple[int, ...]) -> bool:
        return zbin_combo in KiDSLegacy.zbin_combos

    @staticmethod
    def get_ZBcut(zbin_combo: tuple[int, ...]) -> str:
        """Get the ZBcut string for a given zbin combination. This function is
        only used when loading the simulation mass maps."""
        match len(zbin_combo):
            case 1:
                return f"ZBcut{KiDSLegacy._zbin_combo_to_zrange[zbin_combo]}"
            case _:
                raise ValueError("Invalid zbin combination.")


class CosmoSLICS(KiDSLegacy):
    simsname = "KiDSLegacy_CosmoSLICS"
    simspath = "MRres140.64arcs_100Sqdeg_SNKiDS_Mosaic_KiDSLegacyGpAM_zKiDSLegacy_{:s}_Cosmol{:d}"
    simspath_fid = "MRres140.64arcs_100Sqdeg_SNKiDS_Mosaic_KiDSLegacyGpAM_zKiDSLegacy_{:s}_Cosmolfid"
    mapfname = r"SNKiDS_Mosaic.KiDSLegacyGpAM.LOS{}R{:d}.SS2.816.Ekappa.npy"
    cosmol_indices = list(range(25)) + [-1]
    LOS_indices = list(range(1, 26))
    param_ranges = {
        "Omega_m": [0.1, 0.55],
        "S_8": [0.6, 0.9],
        "h": [0.6, 0.9],
        "w_0": [-2.0, 0.5],
    }
    param_texlabels = [r"$\Omega_\mathrm{m}$", r"$S_8$", r"$h$", r"$w_0$"]

    @staticmethod
    def get_sim_massmap(
            cosmol: int,
            zbin_combo: tuple[int, ...],
            region: int,
            LOS: int,
    ) -> NDArray:
        assert (
                CosmoSLICS.has_cosmol(cosmol)
                and KiDSLegacy.has_region(region)
                and KiDSLegacy.has_zbin_combo(zbin_combo)
                and CosmoSLICS.has_LOS(LOS)
        ), "Validation failed for one or more inputs."

        zbcut = KiDSLegacy.get_ZBcut(zbin_combo)

        if cosmol == -1:
            simspath = CosmoSLICS.simspath_fid.format(zbcut)
        else:
            simspath = CosmoSLICS.simspath.format(zbcut, cosmol)

        massmap = np.load(
            os.path.join(
                _data_path, simspath,
                CosmoSLICS.mapfname.format(LOS, region),
            )
        )
        return massmap

    @staticmethod
    def get_fid_massmap(
            zbin_combo: tuple[int, ...],
            region: int,
            LOS: int,
    ) -> NDArray:
        massmap = CosmoSLICS.get_sim_massmap(
            cosmol=-1, zbin_combo=zbin_combo, LOS=LOS, region=region)
        return massmap

    @staticmethod
    def has_LOS(LOS: int) -> bool:
        return 1 <= LOS <= 50

    @staticmethod
    def has_cosmol(cosmol: int) -> bool:
        # SLICS simulation set has 25 wCDM cosmologies and one Lambda-CDM cosmology.
        # -1 points to fiducial cosmology
        return -1 <= cosmol <= 24