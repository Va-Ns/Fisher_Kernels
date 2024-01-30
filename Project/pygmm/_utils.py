import pathlib
from typing import Literal, TypeAlias

import numpy as np
import torch
from pycave import bayes

DataVariant: TypeAlias = Literal["full-sift", "full-rgb", "red-sift", "red-rgb"]


def load_data(variant: DataVariant) -> torch.Tensor:
    """Simple utility function to load the data file.

    Args:
        variant (DataVariant): The variant of the dataset to load.
            Accepted values are: `full-sift`, `full-rgb`, `red-sift`, `red-rgb`.

    Returns:
        torch.Tensor: The loaded data as a tensor.

    """
    variantToName = {
        "full-sift": "sift",
        "full-rgb": "rgb",
        "red-sift": "reduced sift",
        "red-rgb": "reduced rgb",
    }
    return torch.from_numpy(
        np.load(
            str(
                pathlib.Path(__file__).resolve().parent.parent
                / f"data/{variantToName[variant]} feature matrix.npy"
            )
        ).astype(np.float32)
    )


def get_mdl_n_params(mdl: bayes.GaussianMixture) -> int:
    """Return the number of free parameters in the model.

    Adapted from sklearn.mixture.GaussianMixture [1]_.

    Args:
        mdl (bayes.GaussianMixture): GMM Model, fit or unfit.

    Returns:
        int: The number of free parameters.

    References:
    .. [1] `sklearn.mixture.GaussianMixture source code. <https://github.com/scikit-learn/scikit-learn/blob/3f89022fa/sklearn/mixture/_gaussian_mixture.py#L847>`

    """
    _, n_features = mdl.model_.means.shape
    if mdl.covariance_type == "full":
        cov_params = mdl.num_components * n_features * (n_features + 1) // 2
    elif mdl.covariance_type == "diag":
        cov_params = mdl.num_components * n_features
    elif mdl.covariance_type == "tied":
        cov_params = n_features * (n_features + 1) // 2
    elif mdl.covariance_type == "spherical":
        cov_params = mdl.num_components
    mean_params = n_features * mdl.num_components
    return cov_params + mean_params + mdl.num_components - 1
