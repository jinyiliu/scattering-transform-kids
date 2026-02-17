"""
This module merges the libraries from the scatterlens.library module into a
unified library for the purpose of analysing the correlation between distinct
wavelet configurations.
"""
import os
import torch
from typing import Sequence
from numpy.typing import ArrayLike

from scatterlens.library import CovStLibrary
from scatterlens.mcmc import Hartlap_factor


class CovStLibraryCombined:
    def __init__(self, cov_st_libs: list[CovStLibrary]):
        self.cov_st_libs = cov_st_libs
        self.LOS_indices = cov_st_libs[0].sims.LOS_indices
        self.zbin_combos = cov_st_libs[0].sims.zbin_combos

    def calc_sim_scoef(self, zbin_combo: tuple[int, ...], region: int):
        msg = """
        Please use the individual CovStLibrary instances to calculate the
        scattering coefficients for each wavelet configuration separately, and
        then combine them as needed for your analysis.
        """
        raise NotImplementedError(msg)

    def get_sim_scoef(
            self,
            zbin_combo: tuple[int, ...],
            region: int | Sequence[int] | None=None,
            LOS: int | Sequence[int] | None=None,
            region_weights: ArrayLike | str | None="auto",
            j_start: int | None=None,
            j_end: int | None=None,
            isotropic: bool=True,
            drop_S0: bool=True,
            decorrelated_S2: bool=True,
            flatten: bool=True,
            return_type: str="sequence",
    ):
        if return_type == "dict":
            raise NotImplementedError

        sim_scoefs = []
        for cov_st_lib in self.cov_st_libs:
            sim_scoef = cov_st_lib.get_sim_scoef(
                zbin_combo=zbin_combo,
                region=region,
                LOS=LOS,
                region_weights=region_weights,
                j_start=j_start,
                j_end=j_end,
                isotropic=isotropic,
                drop_S0=drop_S0,
                decorrelated_S2=decorrelated_S2,
                flatten=flatten,
                return_type=return_type,
            )
            sim_scoefs.append(sim_scoef)

        return torch.concatenate(sim_scoefs)

    def collect_flattened_scoef(
            self,
            zbin_combos: list[tuple[int, ...]],
            region: int | Sequence[int] | None=None,
            j_start: int | None=None,
            j_end: int | None=None,
            drop_S0: bool=True,
            decorrelated_S2: bool=True,
    ):
        for zbin_combo_ind, zbin_combo in enumerate(zbin_combos):
            for LOS_ind, LOS in enumerate(self.LOS_indices):
                scoef = self.get_sim_scoef(
                    zbin_combo=zbin_combo,
                    region=region,
                    region_weights="auto",
                    LOS=LOS,
                    j_start=j_start,
                    j_end=j_end,
                    drop_S0=drop_S0,
                    isotropic=True,
                    flatten=True,
                    return_type="sequence",
                    decorrelated_S2=decorrelated_S2,
                )
                if "scoef_tensor" not in locals():
                    scoef_tensor = torch.zeros(size=(
                            len(self.LOS_indices),
                            len(zbin_combos),
                            len(scoef),
                    ))
                scoef_tensor[LOS_ind, zbin_combo_ind, :] = scoef

        return scoef_tensor.flatten(start_dim=1, end_dim=2)


    def get_cov(
            self,
            zbin_combos: list[tuple[int]]=None,
            j_start: int | None=None,
            j_end: int | None=None,
            region: int | Sequence[int] | None=None,
            drop_S0: bool=True,
            decorrelated_S2: bool=True,
            Hartlap_correction: bool=False,
            return_mean_dv: bool=False,
            savedir: str | None=None,
            fname: str="cov.pt",
    ):
        scoef_tensor = self.collect_flattened_scoef(
            zbin_combos=zbin_combos if zbin_combos else self.zbin_combos,
            region=region,
            j_start=j_start,
            j_end=j_end,
            drop_S0=drop_S0,
            decorrelated_S2=decorrelated_S2,
        )
        cov = torch.cov(scoef_tensor.t())
        if Hartlap_correction:
            n_dv = scoef_tensor.shape[1]
            n_LOS = scoef_tensor.shape[0]
            cov *= Hartlap_factor(n_LOS, n_dv)

        if savedir:
            torch.save(cov, os.path.join(savedir, fname))

        if return_mean_dv:
            scoef_tensor = torch.mean(scoef_tensor, dim=0)
            return cov, scoef_tensor
        else:
            return cov


class CosmolStLibraryCombined:
    pass