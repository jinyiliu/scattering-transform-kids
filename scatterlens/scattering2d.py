import os
import torch
import numpy as np
from torch.fft import fft2, ifft2
from torch.nn import ZeroPad2d
from torchvision.transforms.functional import resize, InterpolationMode
from numpy.typing import NDArray
from scatterlens.wavelets import Morlet2D

# Part of this code is inspired by Sihao.
# Source: https://github.com/SihaoCheng/scattering_transform

class Scattering2D(object):
    def __init__(
            self, M: int, N: int, J: int, L: int, Q: float=3. / 5. * np.pi,
            sigma_0: float=0.8, dilation_factor: float=2.0, padding: int=0,
            device="cpu", filter_bank: dict[str, torch.Tensor] | str | None=None,
            dtype: torch.dtype=torch.float64, downsample_algo: bool=False,
            return_I1: bool=False):
        """Docstring.

        Args:
            M:
            N:
            J:
            L:
            Q:
            sigma_0:
            dilation_factor:
            padding: The size of zero padding in pixels.
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

        if padding:
            M += 2 * padding
            N += 2 * padding
            self.padding = ZeroPad2d(padding=padding)
        else:
            self.padding = None

        match filter_bank:
            case None:
                filter_bank = Morlet2D(
                    M, N, J, L, Q, sigma_0, dilation_factor,
                ).gen_filter_bank(dtype=dtype)
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
        self.Q = Q
        self.sigma_0 = sigma_0
        self.dilation_factor = dilation_factor


    def scattering(
            self,
            images: torch.Tensor | NDArray,
            large_batch: bool=False,
            mask: torch.Tensor | NDArray | os.PathLike | str | None=None,
            mask_correction: str="fsky",
            local_fsky_min: float=0.1,
            savepath: os.PathLike | str | None=None,
    ):
        """Docstring.

        Args:
            images:
            large_batch:
            mask: A mask for the images where `mask = 1` indicates that the
                pixel is included in the calculation.
            mask_correction: The method to correct for the effect of mask.
                If "fsky", the scattering coefficients will be divided by the
                fraction of sky (fsky) covered by the mask. If "local", the
                scattering coefficients will be calculated locally by dividing
                the scattering maps by the local fraction of sky weighted by the
                modulus of the scattering wavelet. Defaults to "fsky".
            local_fsky_min: Only average over the pixels where the local
                fraction of sky is larger than this threshold. Only applicable
                when `mask_correction` is "local".
            savepath: Path to save the computed scattering coefficients.

        Returns:
            A dict of scattering coefficients.
        """
        assert mask_correction in ("fsky", "local")

        M, N, J, L = self.M, self.N, self.J, self.L
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)

        if images.dim() == 2:
            images = images[None, :, :]
        else:
            assert images.dim() == 3
        num_images = images.shape[0]

        if self.padding:
            images = self.padding(images)

        S0 = torch.zeros((num_images, 1), dtype=self.dtype)
        S1 = torch.zeros((num_images, J, L), dtype=self.dtype)
        S2 = torch.zeros((num_images, J, J, L, L), dtype=self.dtype)

        I1_SET = []

        if self.device == "cuda":
            raise NotImplementedError

        S0[:, 0] = images.mean(dim=(-2, -1))

        mask, fsky = self._read_mask(mask=mask)
        assert mask.shape[-2:] == images.shape[-2:]
        images_f = fft2(images * mask)  # the Fourier of images

        if mask_correction == "local":
            assert not self.downsample_algo, (
                "Local mask correction is not compatible with downsample_algo."
            )
            C1 = ifft2(
                fft2(mask)[None, :, :, :] * fft2(ifft2(self.psi).abs())
            ).real # shape [J, L, M, N]
            C2 = ifft2(
                fft2(C1)[:, None, :, None, :, :] * fft2(ifft2(self.psi).abs())[None, :, None, :, :, :],
            ).real # shape [J, J, L, L, M, N]


        if not large_batch:
            for j1 in range(J):
                if self.downsample_algo:
                    _images_f = _subsample_fourier(images_f, M=M, N=N, j=j1)
                else:
                    _images_f = images_f.clone().detach()

                I1 = ifft2(
                        _images_f[:, None, :, :] * self.psi[j1][None, :, :, :],
                        dim=(-2, -1),
                    ).abs() # shape [num_images, L, M, N]
                M1, N1 = I1.shape[-2:]
                I1 *= (M1 * N1) / (M * N)
                match mask_correction:
                    case "fsky":
                        S1[:, j1] = torch.mean(I1, dim=(-2, -1)) / fsky
                    case "local":
                        I1_ = I1 / C1[j1][None, :, :, :]
                        I1_[:, C1[j1] < local_fsky_min] = float("nan")
                        S1[:, j1] = torch.nanmean(I1_, dim=(-2, -1))
                        del I1_
                I1_f = fft2(I1)
                if self.return_I1:
                    I1_SET.append(I1.mean(dim=1))
                del I1

                for j2 in range(J):
                    if j2 > j1:
                        if self.downsample_algo:
                            _images_f = _subsample_fourier(I1_f, M=M, N=N, j=j2)
                        else:
                            _images_f = I1_f.clone().detach()

                        I2 = ifft2(
                            _images_f[:, :, None, :, :] * self.psi[j2][None, None, :, :, :],
                            dim=(-2, -1),
                        ).abs()
                        M2, N2 = I2.shape[-2:]
                        I2 *= (M2 * N2) / (M1 * N1) # shape [num_images, L, L, M, N]
                        match mask_correction:
                            case "fsky":
                                S2[:, j1, j2, :, :] = torch.mean(I2, dim=(-2, -1)) / fsky
                            case "local":
                                I2_ = I2 / C2[j1][j2][None, :, :, :, :]
                                I2_[:, C2[j1][j2] < local_fsky_min] = float("nan")
                                S2[:, j1, j2, :, :] = torch.nanmean(I2_, dim=(-2, -1))
                                del I2_
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


    def _read_mask(self, mask: torch.Tensor | NDArray | os.PathLike | str | None=None):
        """Read the mask and calculate the fraction of sky (fsky) it covers. If
        mask is None, return an all-ones mask and fksy=1.

        Returns:
            mask: A torch.Tensor of size [1, M, N].
            fsky: A scalar indicating the fraction of sky covered by the mask.
        """
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
                mask = mask[None, :, :]

            if self.padding:
                mask = self.padding(mask)
        else:
            mask = torch.ones((1, self.M, self.N), dtype=self.dtype)

        fsky = torch.mean(mask, dim=(-2, -1)).squeeze()
        return mask, fsky


    def _generate_mask_bank(
            self, mask:torch.Tensor | NDArray | os.PathLike | str | None=None):
        mask, fsky = self._read_mask(mask=mask)
        mask_bank = []
        if self.downsample_algo:
            for j in range(self.J):
                _M, _N = self.psi[j].shape[-2:]
                mask_bank.append(
                    _binary_mask_subsample(mask, size=[_M, _N]))
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






