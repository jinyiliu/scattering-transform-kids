import os
import torch
import numpy as np
from torch.fft import fft2, ifft2
from torchvision.transforms.functional import resize, InterpolationMode
from numpy.typing import NDArray
from scatterlens.wavelets import Morlet2D

# Part of this code is inspired by Sihao.
# Source: https://github.com/SihaoCheng/scattering_transform

class Scattering2D(object):
    def __init__(
            self, M: int, N: int, J: int, L: int, device="cpu",
            filter_bank: dict[str, torch.Tensor] | str | None=None,
            dtype: torch.dtype=torch.float64, downsample_algo: bool=True,
            return_I1: bool=False,):
        """Docstring.

        Args:
            M:
            N:
            J:
            L:
            filter_bank: The selected wavelets to be used for scattering.
                If str, will load the filter_bank using `torch.load`. The dict
                contain two keys: `psi` and `phi`. `psi` corresponds to a
                `torch.Tensor` object with size [J, L, M, N].
            dtype:
            downsample_algo: If True, the j-th scattered image will be
                subsampled to approximately (M / 2**j, N / 2**j). Scattering
                coefficients will be calculated from these downsample images.
                Defaults to False.
            return_I1: If True, the `scattering` method will return the
                first-order scattering maps after averaging over orientations.
        """
        assert device in ("cpu", "cuda"), "Device must be cpu or cuda."

        if device == "cuda":
            raise NotImplementedError
        self.device = device

        self.dtype = dtype
        self.downsample_algo = downsample_algo
        self.return_I1 = return_I1

        match filter_bank:
            case None:
                filter_bank = Morlet2D(M, N, J, L).gen_filter_bank(dtype=dtype)
            case str():
                if os.path.exists(filter_bank):
                    filter_bank = torch.load(filter_bank, weights_only=True)
                else:
                    raise FileNotFoundError(filter_bank)
            case dict():
                assert filter_bank["psi"].shape[:2] == (5, 4)
                assert filter_bank["psi"].dtype == self.dtype
            case _:
                raise ValueError

        self.psi = filter_bank["psi"]
        self.phi = filter_bank["phi"]

        if downsample_algo:
            self.psi = [] # list of torch.Tensor
            for j in range(J):
                _psi = _subsample_fourier(filter_bank["psi"][j], M=M, N=N, j=j)
                self.psi.append(_psi)
        else:
            self.psi = filter_bank["psi"] # torch.Tensor

        self.M = M
        self.N = N
        self.J = J
        self.L = L


    def scattering(
            self, images: torch.Tensor | NDArray, large_batch: bool=False,
            mask: torch.Tensor | NDArray | os.PathLike | str | None=None,
            savepath: os.PathLike | str | None=None,):
        """Docstring.

        Args:
            images:
            large_batch:
            mask: A mask for the images where `mask = 1` indicates that the
                pixel is included in the calculation.
            savepath: Path to save the computed scattering coefficients.

        Returns:
            A dict of scattering coefficients.
        """
        M, N, J, L = self.M, self.N, self.J, self.L
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)

        if images.dim() == 2:
            images = images[None, :, :]
        else:
            assert images.dim() == 3
        num_images = images.shape[0]

        mask_bank = self._generate_mask_bank(mask=mask)

        S0 = torch.zeros((num_images, 1), dtype=self.dtype)
        S1 = torch.zeros((num_images, J, L), dtype=self.dtype)
        S2 = torch.zeros((num_images, J, J, L, L), dtype=self.dtype)

        I1_SET = []

        if self.device == "cuda":
            raise NotImplementedError

        S0[:, 0] = images.mean(dim=(-2, -1))

        images_f = fft2(images) # the Fourier of images

        if not large_batch:
            for j1 in range(J):
                if self.downsample_algo:
                    _images_f = _subsample_fourier(images_f, M=M, N=N, j=j1)
                else:
                    _images_f = images_f.clone().detach()
                mask = mask_bank[j1]
                M1, N1 = mask.shape[-2:]

                I1 = ifft2(
                        _images_f[:, None, :, :] * self.psi[j1][None, :, :, :],
                        dim=(-2, -1),
                    ).abs()
                I1 *= (M1 * N1) / (M * N)
                S1[:, j1] = torch.sum(I1 * mask, dim=(-2, -1)) / mask.sum()
                I1_f = fft2(I1)
                if self.return_I1:
                    I1_SET.append(I1.mean(dim=1) * mask)
                del I1

                for j2 in range(J):
                    if j2 >= j1:
                        if self.downsample_algo:
                            _images_f = _subsample_fourier(I1_f, M=M, N=N, j=j2)
                        else:
                            _images_f = I1_f.clone().detach()
                        mask = mask_bank[j2]
                        M2, N2 = mask.shape[-2:]

                        I2 = ifft2(
                            _images_f[:, :, None, :, :] * self.psi[j2][None, None, :, :, :],
                            dim=(-2, -1),
                        ).abs()
                        I2 *= (M2 * N2) / (M1 * N1)
                        S2[:, j1, j2, :, :] = torch.sum(I2 * mask, dim=(-2, -1)) / mask.sum()
                        del I2
        else:
            raise NotImplementedError

        if self.return_I1:
            return {"S0": S0, "S1": S1, "S2": S2, "I1": I1_SET}
        else:
            ret = {"S0": S0, "S1": S1, "S2": S2}
            if savepath:
                torch.save(ret, savepath)
            else:
                return ret


    def _generate_mask_bank(
            self, mask:torch.Tensor | NDArray | os.PathLike | str | None=None):
        if mask is not None:
            if isinstance(mask, str):
                if mask.endswith(".pt"):
                    mask = torch.load(mask, weights_only=True)
                elif mask.endswith(".npy"):
                    mask = np.load(mask)
                else:
                    raise NotImplementedError

            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)

            if mask.dtype != self.dtype:
                mask = mask.to(dtype=self.dtype)
            assert mask.dim() in (2, 3)

            if mask.dim() == 2:
                assert mask.shape == (self.M, self.N)
                mask = mask[None, :, :]
            mask_bank = []

            if self.downsample_algo:
                for j in range(self.J):
                    _M, _N = self.psi[j].shape[-2:]
                    mask_bank.append(
                        _binary_mask_subsample(mask, size=[_M, _N]))
            else:
                mask_bank = [mask] * self.J
        else:
            mask = torch.ones((1, self.M, self.N), dtype=self.dtype)

            if self.downsample_algo:
                mask_bank = []
                for j in range(self.J):
                    _M, _N = self.psi[j].shape[-2:]
                    mask_bank.append(
                        torch.ones((1, _M, _N), dtype=self.dtype))
            else:
                mask_bank = [mask] * self.J

        return mask_bank




def _cpu2cudaTensor(*args):
    for arg in args:
        arg = arg.cuda()


def _binary_mask_subsample(mask: torch.Tensor, size: list[int]):
    """Subsampling of a 2D binary mask performed in spatial domain."""
    assert mask.dim() == 3
    out = resize(mask, size=size, interpolation=InterpolationMode.NEAREST_EXACT)
    return out


def _subsample_fourier(
        images_fourier: torch.Tensor,
        M: int,
        N: int,
        j: int,
) -> torch.Tensor:
    """Subsampling of a 2D image performed in the Fourier domain.

    Notes:
        This code is copied from the method `cut_high_k_off` in:
        https://github.com/SihaoCheng/scattering_transform/blob/master/scattering/Scattering2d.py
    """
    dx = int( max( 8, min( np.ceil(M/2**j), M//2 )) )
    dy = int( max( 8, min( np.ceil(N/2**j), N//2 )) )

    is_xodd: bool = (images_fourier.shape[-2] % 2 == 1)
    is_yodd: bool = (images_fourier.shape[-1] % 2 == 1)

    out = torch.cat((
            torch.cat((
                images_fourier[..., :dx + is_xodd, :dy + is_yodd],
                images_fourier[..., -dx:, :dy + is_yodd]), dim=-2),
            torch.cat((
                images_fourier[..., :dx + is_xodd, -dy:],
                images_fourier[..., -dx:, -dy:]), dim=-2)
    ), dim=-1)

    return out






