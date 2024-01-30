import math

import torch

try:
    from _utils import get_mdl_n_params
except ImportError:
    from pygmm._utils import get_mdl_n_params

from pycave import bayes


def aic(mdl: bayes.GaussianMixture, data: torch.Tensor) -> float:
    r"""Akaike Information Criterion for a fitted GMM model.

    Adapted from sklearn.mixture.GaussianMixture [1]_. PyCave's GMM score method
    returns the negative log-likelihood unlike sklearn's, which returns
    the log-likelihood. Hence the minus on the term
    :math:`-2 \cdot \ln(\hat{L})`. For whatever reason, the sklearn
    implementation multiplies the first term with the amount of data samples,
    while the definition [2]_ and the corresponding sklearn section [3]_ do not
    that. This implementation coincides with [2]_ and [3]_ and was based on
    [1]_.

    Args:
        mdl (bayes.GaussianMixture): The fitted GMM model.
        data (torch.Tensor): tensor of shape (n_samples, n_features).
            The input samples.

    Returns:
        float: AIC. The lower the better

    References:
        .. [1] `sklearn.mixture.GaussianMixture source code. <https://github.com/scikit-learn/scikit-learn/blob/3f89022fa/sklearn/mixture/_gaussian_mixture.py#L881>`
        .. [2] `Akaike Information Criterion Wikipedia. <https://en.wikipedia.org/wiki/Akaike_information_criterion>`
        .. [3] `scikit-learn Information-criteria based model selection. <https://scikit-learn.org/stable/modules/linear_model.html#information-criteria-based-model-selection>`

    """
    return 2 * mdl.score(data) + 2 * get_mdl_n_params(mdl)


def bic(mdl: bayes.GaussianMixture, data: torch.Tensor) -> float:
    r"""Bayesian Information Criterion for a GMM model.

    Adapted from sklearn.mixture.GaussianMixture [1]_. PyCave's GMM score method
    returns the negative log-likelihood unlike sklearn's, which returns
    the log-likelihood. Hence the minus on the term
    :math:`-2 \cdot \ln(\hat{L})`. For whatever reason, the sklearn
    implementation multiplies the first term with the amount of data samples,
    while the definition [2]_ and the corresponding sklearn section [3]_ do not
    that. This implementation coincides with [2]_ and [3]_ and was based on
    [1]_.

    Args:
        mdl (bayes.GaussianMixture): The fitted GMM model.
        data (torch.Tensor): tensor of shape (n_samples, n_features)
            The input samples.

    Returns:
        float: BIC. The lower the better

    References:
        .. [1] `sklearn.mixture.GaussianMixture source code. <https://github.com/scikit-learn/scikit-learn/blob/3f89022fa/sklearn/mixture/_gaussian_mixture.py#L861>`
        .. [2] `Akaike Information Criterion Wikipedia. <https://en.wikipedia.org/wiki/Akaike_information_criterion>`
        .. [3] `scikit-learn Information-criteria based model selection. <https://scikit-learn.org/stable/modules/linear_model.html#information-criteria-based-model-selection>`

    """
    return 2 * mdl.score(data) + math.log(data.shape[0]) * get_mdl_n_params(mdl)