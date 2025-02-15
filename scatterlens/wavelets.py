import os
import torch
import numpy as np
from scipy.fft import fft2
from numpy.typing import NDArray
from kymatio.scattering2d import filter_bank


class Wavelet2D(object):
    def __init__(self, M: int, N: int, J: int, L: int):
        self.M = M
        self.N = N
        self.J = J # number of scales
        self.L = L # number of angles

    def gen_filter_bank(
            self, dtype: torch.dtype, savedir: str | os.PathLike | None=None,
            overwrite=False):
        """ Define the funtion in child class. """
        raise NotImplementedError


    @staticmethod
    def dilation(j: int) -> int:
        dilation_factor = 2 # binary dilation
        return dilation_factor ** j


class Morlet2D(Wavelet2D):
    def __init__(self, M: int, N: int, J: int, L: int):
        super().__init__(M=M, N=N, J=J, L=L)
        pass

    def morlet_2d(self, j: int, l: int) -> NDArray[np.complex64]:
        """ Return the 2D profile of the Morlet wavelet in real space. """
        psi = filter_bank.morlet_2d(
            M=self.M,
            N=self.N,
            sigma=Morlet2D.sigma(j),
            xi=Morlet2D.k0(j),
            theta=(int(self.L - self.L / 2 - 1) - l) / self.L * np.pi,
            slant=4. / self.L,
            offset=0.,
        )
        return psi

    def gabor_2d(self) -> NDArray[np.complex64]:
        phi = filter_bank.gabor_2d(
            M=self.M,
            N=self.N,
            sigma=Morlet2D.sigma(self.J - 1),
            xi=0.,
            theta=0.,
            slant=1.,
            offset=0.,
        )
        return phi

    def gen_filter_bank(
            self, dtype: torch.dtype, savedir: str | os.PathLike | None=None,
            overwrite=False) -> dict[str, torch.Tensor]:
        """ Generate filter bank in Fourier space. """
        if not dtype in (torch.float32, torch.float64):
            raise AssertionError("dtype must be torch.float32 or torch.float64")

        # Band-pass filters
        psi = torch.zeros(size=(self.J, self.L, self.M, self.N), dtype=dtype)

        for j in range(self.J):
            for l in range(self.L):
                psi_signal = self.morlet_2d(j, l)
                psi_signal_fourier = np.real(fft2(psi_signal))

                # The default precision in numpy is float64
                if dtype==torch.float32:
                    psi_signal_fourier = psi_signal_fourier.astype(np.float32)

                psi[j, l] = torch.from_numpy(psi_signal_fourier)

        # Low-pass filter
        phi_signal = self.gabor_2d()
        phi_signal_fourier =  np.real(fft2(phi_signal))
        if dtype==torch.float32:
            phi_signal_fourier = phi_signal_fourier.astype(np.float32)
        phi = torch.from_numpy(phi_signal_fourier)

        filters = {"psi": psi, "phi": phi}

        if savedir:
            fname = f"Morlet2Dfilters_M{self.M}N{self.N}J{self.J}L{self.L}_{str(dtype).split('.')[-1]}.pt"
            torch.save(filters, os.path.join(savedir, fname))

        return filters

    def get_profile(self):
        pass

    @staticmethod
    def sigma(j):
        """ Sigma of the Gaussian envelope in pixels. """
        base_value: float = 0.8
        return base_value * Wavelet2D.dilation(j)

    @staticmethod
    def k0(j):
        """ Central frequency of the Morlet wavelet with unit per pixel. """
        base_value: float = 3. / 4. * np.pi
        return base_value / Wavelet2D.dilation(j)

    @staticmethod
    def j2scale(j, pixel_length):
        """ Convert j to angular scale in arcmin.
            The parameter pixel_length is arcmin.
        """
        return pixel_length / Morlet2D.k0(j) * 2


def get_Gaussian_profile():
    pass