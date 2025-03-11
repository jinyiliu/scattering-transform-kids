import os
import torch
from glob import glob
from typing import Type

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


class CosmolStLibrary:
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
            self.fname = "SCOEF_{}_Cosmol{}_ZB{}xZB{}_R{}.pt"


    def get_savepath(self, cosmol: int, zbin1: int, zbin2: int, region: int):
        """Return the savepath according to the given region, cosmology, and
        redshift bins."""
        if hasattr(self, "sims"):
            fname = self.fname.format(
                self.sims.simsname, "fid" if cosmol==-1 else cosmol,
                zbin1, zbin2, region)
        else:
            pathname = "*_Cosmol{}_ZB{}xZB{}_R{}.pt".format(
                "fid" if cosmol==-1 else cosmol, zbin1, zbin2, region)
            matched = glob(pathname=pathname, root_dir=self.libdir)
            if not matched:
                raise FileNotFoundError
            else:
                fname = matched[0]

        savepath = os.path.join(self.libdir, fname)
        return savepath


    def calc_sim_scoef(
            self, cosmol: int, zbin1: int, zbin2: int, region: int):
        """Return the scattering coefficients according to the given region,
        cosmology, and redshift bins."""
        if not hasattr(self, "sims"):
            raise AttributeError

        savepath = self.get_savepath(cosmol, zbin1, zbin2, region)

        images = torch.zeros(
            size=(len(self.sims.LOS_indices), *self.sims.region_MN[region]),
        )

        for i, LOS in enumerate(self.sims.LOS_indices):
            mass = self.sims.get_sim_massmap(
                cosmol=cosmol, zbin1=zbin1, zbin2=zbin2, LOS=LOS, region=region)
            images[i] = torch.from_numpy(mass)

        if self.masklib:
            mask = self.masklib.get_savepath(region=region)
        else:
            mask = images[0] != 0.

        self.ST[region].scattering(images, mask=mask, savepath=savepath)


    def get_sim_scoef(self):
        pass

    def get_fid_scoef(self):
        pass

    def get_sim_stats_scoef(self):
        pass

    def get_fid_stats_scoef(self):
        pass




class CovStLibrary:
    def __init__(self, libdir: os.PathLike | str, ):
        pass



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
            pixel_length: float | None=None,):
        """Library to store the masks for different regions."""
        self.libdir = libdir
        if not os.path.exists(self.libdir):
            os.makedirs(self.libdir)

        if sims is not None:
            assert hasattr(sims, "get_fid_massmap")
            assert callable(sims.get_fid_massmap)
            assert hasattr(sims, "region_MN")

            self.fname = "Mask_R{}_M{}N{}.pt"

            for region in sims.region_MN.keys():
                mass = sims.get_fid_massmap(zbin1=1, zbin2=1, LOS=1, region=region)
                mask = mass != 0.
                mask = torch.from_numpy(mask[None, :, :])
                if mask.dtype != dtype:
                    mask = mask.to(dtype)
                torch.save(mask, os.path.join(
                    libdir, self.fname.format(region, *sims.region_MN[region]))
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