"""Python (GPU) GMM.

The `pygmm` package is a configurable and runnable code base that performs
the following:

    1. Trains Gaussian Mixture Model(s) based on the configuration values passed
        in the command line.
    2. Exports the computed GMM parameters (covariances, means and component
        proportions), to a .mat file in order to load them in MATLAB through the
        built-in function `gmdistribution`.

The reason for this decision is because MATLAB's `fitgmdist` is not compatible
with `gpuArrays` hence it cannot leverage CUDA.

This project is not intended for professional use and is only planned out this
way in order for the authors to get more experience in multi-tool development.

"""
import pathlib
from typing import List

import numpy as np
import torch
import tqdm.contrib

try:
    from _config import Settings, get_settings
    from _metrics import aic, bic
    from _utils import load_data
except ImportError:
    from pygmm._config import Settings, get_settings
    from pygmm._metrics import aic, bic
    from pygmm._utils import load_data
from matplotlib import pyplot as plt
from pycave import bayes

import matlab.engine

# Load the dataset
args: Settings = get_settings()
cwd = pathlib.Path.cwd().resolve()
print("\nCurrent working directory:", str(cwd))

if args.data_path is not None:
    data = torch.from_numpy(np.load(args.data_path).astype(np.float32))
    print(
        "Using custom dataset located at",
        str(pathlib.Path(args.data_path).resolve()),
    )
else:
    data = load_data(args.data_variant)
    print(f"Using built-in dataset {args.data_variant}")

print(
    f"Dataset contains {data.shape[0]} samples"
    f" and {data.shape[1]} features."
)

# Set the amount of components to run GMMs for
components: List[int] = [
    args.num_comps,
]
if args.train_multi_gmms:
    components: List[int] = list(range(1, args.num_comps + 1))
# Lists to save the models and the respective metrics.
gmms: List[bayes.GaussianMixture] = []
aic_l: List[float] = []
bic_l: List[float] = []
converged: List[bool] = []

# The main loop over the components
for idx, comp in tqdm.contrib.tenumerate(
    components, desc="Components k", position=0
):
    mdl: bayes.GaussianMixture = bayes.GaussianMixture(
        num_components=comp,
        covariance_type=args.cov_type,
        init_strategy=args.init_strat,
        covariance_regularization=args.cov_regul,
        batch_size=args.batch_size,
        trainer_params={
            "max_epochs": args.max_epochs,
            "accelerator": "gpu",
            "devices": 1,
        },
    )
    mdl.fit(data)
    gmms.append(mdl)
    aic_l.append(aic(mdl, data))
    bic_l.append(bic(mdl, data))
    converged.append(mdl.converged_)
del data

# Connects to an existing and shared MATLAB session, which is done by
# running `matlab.engine.shareEngine(Name)` in MATLAB.
if hasattr(args, "matlab_session"):
    eng = matlab.engine.connect_matlab(args.matlab_session)
else:  # else starts a new session
    eng = matlab.engine.start_matlab()
eng.cd(str(pathlib.Path(__file__).resolve().parent.parent / "matlab"))

covs: List[torch.Tensor] = []
means: List[torch.Tensor] = []
prop: List[List[float]] = []
for gmm in gmms:
    covs.append(gmm.model_.covariances.T)
    means.append(gmm.model_.means)
    prop.append(gmm.model_.component_probs)

covs: torch.Tensor = torch.cat(covs, dim=1)
means: torch.Tensor = torch.cat(means, dim=0)
prop: torch.Tensor = torch.cat(prop)

save_path = (cwd / args.output_dir).resolve()
save_path.mkdir(parents=True, exist_ok=True)

eng.saveFromPython(
    str(save_path),
    eng.reshape(
        eng.double(covs.cpu().numpy()), 1, covs.shape[0], covs.shape[1]
    ),
    eng.double(means.cpu().numpy()),
    eng.double(prop.cpu().numpy()),
    nargout=0,
)

eng.exit()  # This does nothing if a MATLAB session name was passed.

if args.gmm_plot_range < 0:
    exit(0)

min_aic_idx: int = aic_l.index(min(aic_l))
slice_range = min_aic_idx
title = f"Min AIC at $k$ = {min_aic_idx + 1}"
if args.gmm_plot_range > 1 and len(components) > 1:
    start: int = min_aic_idx - args.gmm_plot_range
    start = start if start >= 0 else 0
    end: int = min_aic_idx + args.gmm_plot_range
    end = end if end <= len(components) else len(components)
    slice_range = slice(start, end)
    title = (
        f"AIC for $k$ = {start+1}:{end+1}, min AIC at $k$ = {min_aic_idx + 1}"
    )

plt.figure(1)
plt.bar(components[slice_range], aic_l[slice_range])
plt.title(title)
plt.xlabel("$k$")
plt.ylabel("AIC")
# we don't block this in order for the 2nd figure to show up
plt.show(block=False)

min_bic_idx: int = bic_l.index(min(bic_l))
slice_range = min_bic_idx
title = f"Min BIC at $k$ = {min_bic_idx + 1}"
if args.gmm_plot_range > 1 and len(components) > 1:
    start: int = min_bic_idx - args.gmm_plot_range
    start = start if start >= 0 else 0
    end: int = min_bic_idx + args.gmm_plot_range
    end = end if end <= len(components) else len(components)
    slice_range = slice(start, end)
    title = (
        f"BIC for $k$ = {start+1}:{end+1}, min AIC at $k$ = {min_bic_idx + 1}"
    )

plt.figure(2)
plt.bar(components[slice_range], bic_l[slice_range])
plt.title(title)
plt.xlabel("$k$")
plt.ylabel("BIC")
# we block in order for the script to not close the figures from exiting
plt.show(block=True)
