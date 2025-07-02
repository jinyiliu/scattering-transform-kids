import torch
import numpy as np
from copy import deepcopy
from typing import override
from sklearn.base import BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression as LinearRegressor

# Part of this code is copied from Lars.
# Source: https://github.com/ljabbo/kids-persistent-homology


kernels = [RBF,]

class Emulator:
    def __init__(
            self,
            training_set: dict[str, torch.Tensor],
            input_scaler=None,
            target_scaler=None,
            regressor_type=GaussianProcessRegressor,
            **regressor_args,
    ):
        assert "input" in training_set.keys()
        assert "target" in training_set.keys()

        self.input_scaler = input_scaler
        self.target_scaler = target_scaler

        self.regressor_type = regressor_type
        self.regressor_args = regressor_args
        self.regressor = regressor_type(**regressor_args)
        self.is_fitted = False

        self._training_set = {
            "input": np.array(training_set["input"]),
            "target": np.array(training_set["target"]),
        }

        self.training_set = deepcopy(self._training_set)

        for scaler, key in zip(
                [self.input_scaler, self.target_scaler], ["input", "target"]
        ):
            if scaler is not None:
                self.training_set[key] = scaler.fit_transform(self._training_set[key])

        self.n_samples = self.training_set["input"].shape[0]
        self.n_parameters = self.training_set["input"].shape[1]
        self.n_features = self.training_set["target"].shape[1]


    def fit(self):
        self.regressor.fit(self.training_set["input"], self.training_set["target"])
        self.is_fitted = True

    def predict(self, X):
        return self._predict(self.regressor, X)

    def validation(self):
        loo = LeaveOneOut()

        all_mse = []
        all_mse_frac = []

        for i, (train_index, test_index) in enumerate(loo.split(self.training_set["input"])):
            temp_regressor = self.regressor_type(**self.regressor_args)
            temp_regressor.fit(
                self.training_set["input"][train_index],
                self.training_set["target"][train_index],
            )
            mse = (
                self._training_set["target"][test_index][0] -
                self._predict(temp_regressor, X=self.training_set["input"][test_index])[0]
            )
            mse_frac = mse / self._training_set["target"][test_index][0]

            all_mse.append(mse)
            all_mse_frac.append(mse_frac)

        avg_mse = np.nanmean(all_mse, axis=0)
        avg_mse_frac = np.nanmean(all_mse_frac, axis=0)
        return avg_mse, all_mse, avg_mse_frac, all_mse_frac

    def _predict(self, regressor, X):
        if self.input_scaler is not None:
            X = self.input_scaler.transform(X)

        Y = regressor.predict(X)
        if self.target_scaler is not None:
            Y = self.target_scaler.inverse_transform(Y)

        return Y


class S1S2Emulator(Emulator):
    def __init__(
            self,
            training_set: dict[str, torch.Tensor],
            J: int,
            per_zbincombo_emulator: bool=False,
            regressor_type=GaussianProcessRegressor,
            **regressor_args,
    ):
        """

        Args:
            training_set:
            J: Number of scattering scales
            per_zbincombo_emulator: If True, trains separate emulators for each
                redshift bin combination (e.g., (1,1), (1,2), (1,2,3), ...).
                Defaults to False.
            regressor_type:
            **regressor_args:
        """
        super().__init__(training_set, regressor_type, **regressor_args)
        self._J = J
        if self.n_features % (self.n_S1 + self.n_S2) != 0:
            raise ValueError
        self.n_zbincombo = self.n_features // (self.n_S1 + self.n_S2)
        self.per_zbincombo_emulator = per_zbincombo_emulator

        self.n_regressors = 2
        if self.per_zbincombo_emulator:
            self.n_regressors *= self.n_zbincombo

        self.regressors = self._create_regressors()


    @property
    def n_S1(self):
        return self._J

    @property
    def n_S2(self):
        return (self._J * (self._J + 1)) // 2

    @override
    def fit(self):
        self.regressors = self._fit(self.regressors, self.training_set)

    @override
    def predict(self, X):
        return self._predict(regressors=self.regressors, X=X)

    @override
    def validation(self):
        loo = LeaveOneOut()

        all_mse = []
        all_mse_frac = []

        for i, (train_index, test_index) in enumerate(
                loo.split(self.training_set["input"])):
            temp_regressors = self._create_regressors()
            temp_training_set = {
                "input": self.training_set["input"][train_index],
                "target": self.training_set["target"][train_index],
            }
            self._fit(temp_regressors, temp_training_set)

            mse = (
                self.training_set["target"][test_index][0] -
                self._predict(
                    regressors=temp_regressors,
                    X=self.training_set["input"][test_index])[0]
            )
            mse_frac = mse / self.training_set["target"][test_index][0]

            all_mse.append(mse)
            all_mse_frac.append(mse_frac)

        avg_mse = np.nanmean(all_mse, axis=0)
        avg_mse_frac = np.nanmean(all_mse_frac, axis=0)
        return avg_mse, all_mse, avg_mse_frac, all_mse_frac


    def _create_regressors(self) -> list[BaseEstimator] | list[list[BaseEstimator]]:
        """Instantiate regressors based on current configuration."""
        # FIXME all regressors point to the same memory id
        if self.per_zbincombo_emulator:
            regressors = [
                [
                    self.regressor_type(**self.regressor_args)
                    for combo in range(self.n_zbincombo)
                ]
                for _ in range(2)
            ]
        else:
            regressors = [
                self.regressor_type(**self.regressor_args) for _ in range(2)
            ]
        return regressors


    def _fit(self, regressors, training_set):
        # Reshape the target feature vector into size
        # (n_samples, n_zbincombo, n_S1 + n_S2)
        target_array = training_set["target"].reshape(
            -1, self.n_zbincombo, self.n_S1 + self.n_S2
        )
        n_samples = target_array.shape[0]
        if self.per_zbincombo_emulator:
            trained_regressors = [
                [
                    regressor.fit(
                        training_set["input"], target_array[:, i, :self.n_S1]
                    ) for i, regressor in enumerate(regressors[0])
                ],
                [
                    regressor.fit(
                        training_set["input"], target_array[:, i, self.n_S1:]
                    ) for i, regressor in enumerate(regressors[1])
                ],
            ]
        else:
            S1S2 = [
                target_array[:, :, :self.n_S1].reshape(n_samples, -1),
                target_array[:, :, self.n_S1:].reshape(n_samples, -1),
            ]
            trained_regressors = [
                regressor.fit(training_set["input"], target)
                for target, regressor in zip(S1S2, regressors)
            ]
        return trained_regressors


    def _predict(self, regressors, X):
        if self.per_zbincombo_emulator:
            S1 = np.vstack([
                regressor.predict(X)[0] for regressor in regressors[0]
            ]).reshape(len(X), self.n_zbincombo, self.n_S1)
            S2 = np.vstack([
                regressor.predict(X)[0] for regressor in regressors[1]
            ]).reshape(len(X), self.n_zbincombo, self.n_S2)
        else:
            # FIXME only works for one set of input parameters
            S1 = regressors[0].predict(X).reshape(len(X), self.n_zbincombo, self.n_S1)
            S2 = regressors[1].predict(X).reshape(len(X), self.n_zbincombo, self.n_S2)

        S1S2 = np.concatenate((S1, S2), axis=-1).reshape(len(X), self.n_features)
        return S1S2



class PerFeatureEmulator(Emulator):
    def __init__(
            self,
            training_set: dict[str, torch.Tensor],
            input_scaler=None,
            target_scaler=None,
            regressor_type=GaussianProcessRegressor,
            **regressor_args,
    ):
        super().__init__(
            training_set, input_scaler, target_scaler, regressor_type, **regressor_args
        )

        self.regressors = [
            regressor_type(**regressor_args) for _ in range(self.n_features)
        ]

    @override
    def fit(self):
        self.regressors = self._fit(self.regressors, self.training_set)
        self.is_fitted = True

    @staticmethod
    def _fit(regressors, training_set):
        trained_regressors = [
            regressor.fit(
                training_set["input"], training_set["target"][:, i]
            )
            for i, regressor in enumerate(regressors)
        ]
        return trained_regressors

    @override
    def predict(self, X):
        return self._predict(self.regressors, X)

    @override
    def _predict(self, regressors, X):
        # FIXME only works for one set of input parameters
        if self.input_scaler is not None:
            X = self.input_scaler.transform(X)

        Y = np.array([[
            regressor.predict(X)[0] for regressor in regressors
        ]])
        if self.target_scaler is not None:
            Y = self.target_scaler.inverse_transform(Y)

        return Y

    @override
    def validation(self):
        loo = LeaveOneOut()

        all_mse = []
        all_mse_frac = []

        for i, (train_index, test_index) in enumerate(
                loo.split(self.training_set["input"])):
            temp_regressors = [
                self.regressor_type(**self.regressor_args)
                for _ in range(self.n_features)
            ]
            temp_training_set = {
                "input": self.training_set["input"][train_index],
                "target": self.training_set["target"][train_index],
            }
            temp_regressors = self._fit(temp_regressors, temp_training_set)
            mse = (
                self._training_set["target"][test_index][0] -
                self._predict(temp_regressors, X=self.training_set["input"][test_index])[0]
            )

            mse_frac = mse / self._training_set["target"][test_index][0]

            all_mse.append(mse)
            all_mse_frac.append(mse_frac)

        avg_mse = np.nanmean(all_mse, axis=0)
        avg_mse_frac = np.nanmean(all_mse_frac, axis=0)
        return avg_mse, all_mse, avg_mse_frac, all_mse_frac


class PolynomialRegressor:
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self, X):
        pass