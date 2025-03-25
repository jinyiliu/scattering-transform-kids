import os
import torch
from numpy.typing import ArrayLike
import pandas as pd
from glob import glob
from typing import Sequence

from scatterlens.scattering2d import Scattering2D
from scatterlens.wavelets import Morlet2D


def _get_Scattering2D_per_region(
        region_MN: dict[int, tuple[int, int]], filterlib=None, **kwargs
) -> dict[int, Scattering2D]:
    """Return a dict of Scattering2D instances for each region.

    Args:
        region_MN: The image size of each region.
        filterlib: The scatterlens.library.FilterLibrary instance.
        **kwargs: The keyword arguments for initialising Scattering2D.
    """
    ret = {}
    for region in region_MN.keys():
        M, N = region_MN[region]
        if filterlib:
            filter_bank = filterlib.get_savepath(region=region)
        else:
            filter_bank = None
        ret[region] = Scattering2D(M=M, N=N, filter_bank=filter_bank, **kwargs)
    return ret


class _StLibrary:
    def __init__(
            self,
            libdir: os.PathLike | str,
            filterlib=None,
            masklib=None,
            sims=None,
            **kwargs):
        """Library to store the calculated coefficients for different
        cosmologies.

        Args:
            libdir: Directory to store the calculated coefficients.
            filterlib: The scatterlens.library.FilterLibrary instance. If None,
                will calculate the filters on the fly.
            masklib: The scatterlens.library.MaskLibrary instance. If None, will
                calculate the mask on the fly.
            sims: Simulation class.
            **kwargs: The keyword arguments for initialising Scattering2D.
        """
        self.libdir = libdir
        self.filterlib = filterlib
        self.masklib = masklib
        if not os.path.exists(self.libdir):
            os.makedirs(self.libdir)

        if sims is not None:
            self.sims = sims
            self.masklib = masklib
            assert hasattr(sims, "get_sim_massmap")
            assert callable(sims.get_sim_massmap)
            assert hasattr(sims, "LOS_indices")
            assert hasattr(sims, "simsname")
            assert hasattr(sims, "region_MN")

            self.ST = _get_Scattering2D_per_region(
                sims.region_MN, filterlib=filterlib, **kwargs)


    def get_savepath(self, **kwargs):
        raise NotImplementedError("Define this function in the child class.")


    def calc_sim_scoef(self, **kwargs):
        """Calculate the scattering coefficients according to the given region,
        cosmology, and redshift bins."""
        if not hasattr(self, "sims"):
            raise AttributeError

        savepath = self.get_savepath(**kwargs)

        region = kwargs["region"]

        images = torch.zeros(
            size=(len(self.sims.LOS_indices), *self.sims.region_MN[region]),
        )

        for i, LOS in enumerate(self.sims.LOS_indices):
            mass = self.sims.get_sim_massmap(**kwargs, LOS=LOS)
            images[i] = torch.from_numpy(mass)

        if self.masklib:
            mask = self.masklib.get_savepath(region=region)
        else:
            mask = images[0] != 0.

        self.ST[region].scattering(images, mask=mask, savepath=savepath)


    def _get_sim_scoef_from_paths(
            self,
            st_paths: Sequence[os.PathLike | str],
            LOS: Sequence[int] | None=None,
            region_weights: ArrayLike | str | None="auto",
            j_start: int | None=None,
            j_end: int | None=None,
            j2_equals_j1: bool=False,
            isotropic: bool=True,
            drop_S0: bool=True,
            decorrelated_S2: bool=True,
            flatten: bool = True,
            return_type: str = "dict",
    ) -> dict | Sequence:
        assert return_type in ("dict", "sequence")
        if return_type == "sequence" and not flatten:
            flatten = True

        if region_weights:
            if region_weights == "auto":
                assert hasattr(self, "masklib")
                region = [int(st_path[-5:-3].lstrip("R")) for st_path in st_paths]
                region_weights = self.masklib.get_region_areas(region)
            else:
                assert len(region_weights) == len(st_paths)
        else:
            region_weights = [1.] * len(st_paths)


        if LOS:
            assert hasattr(self, "sims")
            if isinstance(LOS, int):
                LOS = [LOS]
            st_indices = [self.sims.LOS_indices.index(_LOS) for _LOS in LOS]
        else:
            st_indices = slice(None)

        coef = torch.load(
            os.path.join(self.libdir, st_paths[0]), weights_only=True)

        J, L = coef["S1"].shape[-2:]
        S0 = torch.zeros(size=(1, ))
        S1 = torch.zeros(size=(J, L))
        S2 = torch.zeros(size=(J, J, L, L))

        for st_path, region_weight in zip(st_paths, region_weights):
            coef = torch.load(
                os.path.join(self.libdir, st_path), weights_only=True)
            norm = region_weight / sum(region_weights)
            S0 += coef["S0"][st_indices].mean(dim=0) * norm
            S1 += coef["S1"][st_indices].mean(dim=0) * norm
            S2 += coef["S2"][st_indices].mean(dim=0) * norm

        end = j_end + 1 if j_end else None
        S1 = S1[j_start:end]
        S2 = S2[j_start:end, j_start:end]

        if j2_equals_j1:
            S2 = S2.diagonal(dim1=0, dim2=1).permute(2, 1, 0)

        if isotropic:
            S1 = S1.mean(dim=-1)
            S2 = S2.mean(dim=(-2, -1))
            if decorrelated_S2:
                S2 = S2 / S1
        else:
            if decorrelated_S2:
                raise NotImplementedError

        if flatten:
            S0 = S0.flatten()
            S1 = S1.flatten()
            S2 = S2.flatten()
            S2 = S2[S2 != 0.]

        if return_type == "dict":
            if drop_S0:
                return {"S1": S1, "S2": S2}
            else:
                return {"S0": S0, "S1": S1, "S2": S2}

        if return_type == "sequence":
            if drop_S0:
                return torch.hstack((S1, S2))
            else:
                return torch.hstack((S0, S1, S2))


    def find_fname(self, fname_patt: str):
        """Find the filename according to the given filename pattern."""
        matched = glob(pathname=fname_patt, root_dir=self.libdir)
        if not matched:
            raise FileNotFoundError
        else:
            fname = matched[0]
        return fname





class CosmolStLibrary(_StLibrary):
    def __init__(
            self,
            libdir: os.PathLike | str,
            filterlib=None,
            masklib=None,
            sims=None,
            **kwargs):
        super().__init__(libdir, filterlib, masklib, sims, **kwargs)
        self.fname = "SCOEF_{}_Cosmol{}_ZB{}xZB{}_R{}.pt"


    def get_savepath(self, cosmol: int, zbin1: int, zbin2: int, region: int):
        """Return the savepath according to the given region, cosmology, and
        redshift bins."""
        if hasattr(self, "sims"):
            fname = self.fname.format(
                self.sims.simsname, "fid" if cosmol==-1 else cosmol,
                zbin1, zbin2, region)
        else:
            fname_patt = "*_Cosmol{}_ZB{}xZB{}_R{}.pt".format(
                "fid" if cosmol==-1 else cosmol, zbin1, zbin2, region)
            fname = self.find_fname(fname_patt=fname_patt)

        savepath = os.path.join(self.libdir, fname)
        return savepath


    def calc_sim_scoef(
            self, cosmol: int, zbin1: int, zbin2: int, region: int):
        return super().calc_sim_scoef(
            cosmol=cosmol, zbin1=zbin1, zbin2=zbin2, region=region)


    def get_sim_scoef(
            self,
            cosmol: int | str,
            zbin1: int,
            zbin2: int,
            region: int | Sequence[int] | None=None,
            LOS: int | Sequence[int] | None=None,
            region_weights: ArrayLike | str | None="auto",
            j_start: int | None=None,
            j_end: int | None=None,
            j2_equals_j1: bool=False,
            isotropic: bool=True,
            drop_S0: bool=True,
            decorrelated_S2: bool=True,
            flatten: bool=True,
            return_type: str="dict",
    ):
        """Return the scattering coefficients according to the given region,
        cosmology, redshift bins, and LOS."""
        assert zbin2 >= zbin1

        st_paths = []
        if region:
            if isinstance(region, int):
                region = [region]
            for _region in region:
                st_paths.append(self.get_savepath(cosmol, zbin1, zbin2, _region))
        else:
            pathname = "*_Cosmol{}_ZB{}xZB{}_*.pt".format(
                "fid" if cosmol == -1 else cosmol, zbin1, zbin2)
            st_paths = glob(pathname=pathname, root_dir=self.libdir)
            if len(st_paths) == 0:
                raise FileNotFoundError

        return super()._get_sim_scoef_from_paths(
            st_paths=st_paths,
            LOS=LOS,
            region_weights=region_weights,
            j_start=j_start,
            j_end=j_end,
            j2_equals_j1=j2_equals_j1,
            isotropic=isotropic,
            drop_S0=drop_S0,
            decorrelated_S2=decorrelated_S2,
            flatten=flatten,
            return_type=return_type,
        )




    def get_fid_scoef(
            self,
            zbin1: int,
            zbin2: int,
            region: int | Sequence[int] | None=None,
            LOS: int | Sequence[int] | None=None,
    ):
        return self.get_sim_scoef(
            cosmol="fid", zbin1=zbin1, zbin2=zbin2, region=region, LOS=LOS)



class CovStLibrary(_StLibrary):
    def __init__(
            self,
            libdir: os.PathLike | str,
            filterlib=None,
            masklib=None,
            sims=None,
            **kwargs):
        super().__init__(libdir, filterlib, masklib, sims, **kwargs)
        self.fname = "SCOEF_{}_ZB{}xZB{}_R{}.pt"


    def get_savepath(self, zbin1: int, zbin2: int, region: int):
        """Return the savepath according to the given region and redshift bins."""
        if hasattr(self, "sims"):
            fname = self.fname.format(self.sims.simsname, zbin1, zbin2, region)
        else:
            fname_patt = "*_ZB{}xZB{}_R{}.pt".format(zbin1, zbin2, region)
            fname = self.find_fname(fname_patt=fname_patt)

        savepath = os.path.join(self.libdir, fname)
        return savepath


    def calc_sim_scoef(self, zbin1: int, zbin2: int, region: int):
        """Calculate the scattering coefficients according to the given region
        and redshift bins."""
        return super().calc_sim_scoef(zbin1=zbin1, zbin2=zbin2, region=region)


    def get_sim_scoef(
            self,
            zbin1: int,
            zbin2: int,
            region: int | Sequence[int] | None=None,
            LOS: int | Sequence[int] | None=None,
            region_weights: ArrayLike | str | None="auto",
            j_start: int | None=None,
            j_end: int | None=None,
            j2_equals_j1: bool=False,
            isotropic: bool=True,
            drop_S0: bool=True,
            decorrelated_S2: bool=True,
            flatten: bool=True,
            return_type: str="dict",
    ):
        """Return the scattering coefficients according to the given region,
        cosmology, redshift bins, and LOS."""
        assert zbin2 >= zbin1

        st_paths = []
        if region:
            if isinstance(region, int):
                region = [region]
            for _region in region:
                st_paths.append(self.get_savepath(zbin1, zbin2, _region))
        else:
            pathname = "*_ZB{}xZB{}_*.pt".format(zbin1, zbin2)
            st_paths = glob(pathname=pathname, root_dir=self.libdir)
            if len(st_paths) == 0:
                raise FileNotFoundError

        return super()._get_sim_scoef_from_paths(
            st_paths=st_paths,
            LOS=LOS,
            region_weights=region_weights,
            j_start=j_start,
            j_end=j_end,
            j2_equals_j1=j2_equals_j1,
            isotropic=isotropic,
            drop_S0=drop_S0,
            decorrelated_S2=decorrelated_S2,
            flatten=flatten,
            return_type=return_type,
        )

    def get_cov(
            self,
            zbin_pairs: list[tuple[int, int]]=None,
            j_start: int | None=None,
            j_end: int | None=None,
            j2_equals_j1: bool=False,
            region: int | Sequence[int] | None=None,
            drop_S0: bool=True,
            decorrelated_S2: bool=True,
            norm: bool=True,
    ):
        """Return the covariance matrix."""
        assert hasattr(self, "sims")

        for zpair_ind, (zbin1, zbin2) in enumerate(zbin_pairs):
            for LOS_ind, LOS in enumerate(self.sims.LOS_indices):
                scoef = self.get_sim_scoef(
                    zbin1=zbin1,
                    zbin2=zbin2,
                    region=region,
                    region_weights="auto",
                    LOS=LOS,
                    j_start=j_start,
                    j_end=j_end,
                    j2_equals_j1=j2_equals_j1,
                    drop_S0=drop_S0,
                    isotropic=True,
                    flatten=True,
                    return_type="sequence",
                    decorrelated_S2=decorrelated_S2,
                )
                if "scoef_tensor" not in locals():
                    scoef_tensor = torch.zeros(size=(
                            len(zbin_pairs),
                            len(scoef),
                            len(self.sims.LOS_indices),
                    ))
                scoef_tensor[zpair_ind, :, LOS_ind] = scoef

        scoef_tensor = scoef_tensor.flatten(start_dim=0, end_dim=1)

        if norm:
            return torch.corrcoef(scoef_tensor)
        else:
            return torch.cov(scoef_tensor)


class FilterLibrary:
    def __init__(
            self, libdir: os.PathLike | str,
            J: int | None=None, L: int | None=None, dtype: torch.dtype | None=None,
            region_MN: dict[int, tuple[int, int]] | None=None):
        """Library to store the filters in different regions."""
        self.libdir = libdir
        if not os.path.exists(self.libdir):
            os.makedirs(self.libdir)

        if region_MN is not None:
            if J and L and dtype:
                self.J = J
                self.L = L
                self.fname = "Morlet2Dfilters_R{}_M{}N{}J{}L{}_{}.pt"
                for region in region_MN.keys():
                    M, N = region_MN[region]
                    fb = Morlet2D(M, N, J, L).gen_filter_bank(dtype=dtype)
                    torch.save(fb, os.path.join(
                        libdir, self.fname.format(
                            region, M, N, J, L, str(dtype).split('.')[-1]))
                    )
            else:
                raise AttributeError("J and L must not be None when `region_MN` is given.")


    def get_savepath(self, region: int):
        pathname = f"*_R{region}_*.pt"
        matched = glob(pathname=pathname, root_dir=self.libdir)
        if not matched:
            raise FileNotFoundError
        else:
            fname = matched[0]

        savepath = os.path.join(self.libdir, fname)
        return savepath



class MaskLibrary:
    def __init__(
            self, libdir: os.PathLike | str,
            dtype: torch.dtype | None=None,
            sims=None,
    ):
        """Library to store the masks for different regions."""
        self.libdir = libdir
        if not os.path.exists(self.libdir):
            os.makedirs(self.libdir)

        if sims is not None:
            assert hasattr(sims, "get_fid_massmap")
            assert callable(sims.get_fid_massmap)
            assert hasattr(sims, "region_MN")
            assert hasattr(sims, "resol")

            self.fname = "Mask_R{}_M{}N{}.pt"

            pixel_area = (sims.resol / 60) ** 2 # square degree
            df = pd.DataFrame.from_dict(
                sims.region_MN, orient='index', columns=['M', 'N'])
            sky_area = []

            for region in sims.region_MN.keys():
                mass = sims.get_fid_massmap(zbin1=1, zbin2=1, LOS=1, region=region)
                mask = mass != 0.

                sky_area.append(mask.sum() * pixel_area)

                mask = torch.from_numpy(mask[None, :, :])
                if mask.dtype != dtype:
                    mask = mask.to(dtype)
                torch.save(mask, os.path.join(
                    libdir, self.fname.format(region, *sims.region_MN[region]))
                )

            df["Area[deg2]"] = sky_area
            df.to_csv(
                os.path.join(self.libdir, "Sky_Areas.txt"),
                sep="\t", index=True, header=True, float_format="%.6f",
            )

        self._areas = pd.read_csv(
            os.path.join(self.libdir, "Sky_Areas.txt"),
            sep="\t", index_col=0,
        )


    def get_savepath(self, region: int):
        pathname = f"*_R{region}_*.pt"
        matched = glob(pathname=pathname, root_dir=self.libdir)
        if not matched:
            raise FileNotFoundError
        else:
            fname = matched[0]

        savepath = os.path.join(self.libdir, fname)
        return savepath

    def get_region_areas(self, region: Sequence[int] | int):
        ret = []
        for _region in region:
            ret.append(float(self._areas.loc[_region]["Area[deg2]"]))
        return ret