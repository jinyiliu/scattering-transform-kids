import torch
import numpy as np
from typing import Callable

from scatterlens.emulator import Emulator

class Model:
    def __init__(
            self,
            emulator: Emulator,
            systematics: list[str] | None=None,
            IA_kwargs: dict | None = None,
    ):
        """Forward model the statistics with systematic effects.

        Args:
            emulator:
        """
        self.emulator = emulator

        if systematics:
            if not any(syst in ("IA",) for syst in systematics):
                raise ValueError
        self.systematics = systematics

        if IA_kwargs:
            self.IA_kwargs = IA_kwargs
            self.IA_poly_coefs = np.array(self._load_IA_poly_coefs())
            self.IA_fid = self._get_IA_fid()


    def predict(self, params) -> np.ndarray:
        dv = self.emulator.predict(params[:4])
        if self.systematics:
            for syst, param in zip(self.systematics, params[4:]):
                match syst:
                    case "IA":
                        dv *= self.IA(param)
                    case _:
                        pass
        return dv



    def IA(self, A_IA) -> np.ndarray:
        """Intrinsic alignment model.

        Args:
            A_IA: Amplitude of the intrinsic alignment signal.
        """
        degree = self.IA_poly_coefs.shape[0] - 1
        X = np.vander(x=[A_IA], N=degree + 1, increasing=False)
        IA_scaling = (X @ self.IA_poly_coefs).squeeze(0) / self.IA_fid
        return IA_scaling


    def _get_IA_fid(self):
        A_IA = 0.
        degree = self.IA_poly_coefs.shape[0] - 1
        X = np.vander(x=[A_IA], N=degree + 1, increasing=False)
        IA_fid = (X @ self.IA_poly_coefs).squeeze(0)
        return IA_fid


    def _load_IA_poly_coefs(self) -> torch.Tensor:
        assert "savepath" in self.IA_kwargs
        return torch.load(self.IA_kwargs["savepath"], weights_only=True)


    @staticmethod
    def mbias_correction(
            mbias_uncertainties_zbin,
            mean_mbias_zbin,
            zbin_combos,
            galaxy_weighting_zbin=None,
    ):
        """
        Perform a multiplicative bias correction to the scattering coefficients.

        Args:
            mbias_uncertainties_zbin: Nusance parameter for m-bias uncertainty
                in each redshift bin.
            mean_mbias_zbin: The mean m-bias in each redshift bin.
            zbin_combos:
            galaxy_weighting_zbin:
        """
        pass