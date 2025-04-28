import torch
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import LeaveOneOut

# Part of this code is copied from Lars.
# Source: https://github.com/ljabbo/kids-persistent-homology


kernels = [RBF,]

class Emulator:
    def __init__(
            self,
            training_set: dict[str, torch.Tensor],
            regressor_type=GaussianProcessRegressor,
            **regressor_args,
    ):
        assert "input" in training_set.keys()
        assert "target" in training_set.keys()

        self.regressor_type = regressor_type
        self.regressor_args = regressor_args
        self.regressor = regressor_type(**regressor_args)

        self.training_set = {
            "input": np.array(training_set["input"]),
            "target": np.array(training_set["target"]),
        }


    def fit(self):
        self.regressor.fit(self.training_set["input"], self.training_set["target"])

    def predict(self, X):
        return self.regressor.predict(X)

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
                self.training_set["target"][test_index][0] -
                temp_regressor.predict(self.training_set["input"][test_index])[0]
            )
            mse_frac = (
                self.training_set["target"][test_index][0] -
                temp_regressor.predict(self.training_set["input"][test_index])[0]
            ) / self.training_set["target"][test_index][0]

            all_mse.append(mse)
            all_mse_frac.append(mse_frac)

        avg_mse = np.nanmean(all_mse, axis=0)
        avg_mse_frac = np.nanmean(all_mse_frac, axis=0)
        return avg_mse, all_mse, avg_mse_frac, all_mse_frac
