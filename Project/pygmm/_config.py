import argparse
from typing import Protocol

from pycave import bayes


class Settings(Protocol):
    num_comps: int
    train_multi_gmms: bool
    cov_type: bayes.core.CovarianceType
    init_strat: bayes.gmm.types.GaussianMixtureInitStrategy
    batch_size: int
    max_epochs: int
    cov_regul: float
    data_variant: str
    data_path: str
    matlab_session: str
    gmm_plot_range: int
    output_dir: str


def get_settings() -> Settings:
    parser = argparse.ArgumentParser(
        prog="Python (GPU) GMM",
        description=(
            "A runnable and configurable training for GMM(s) using GPU"
            " accelaration."
        ),
    )

    parser.add_argument(
        "-k",
        "--num-comps",
        type=int,
        default=128,
        help="The amount of Gaussian's to fit in a GMM. Defaults to 128",
    )
    parser.add_argument(
        "-t",
        "--train-multi-gmms",
        action="store_true",
        help=(
            "Specifies if multiple GMM(s) are trained going from k=1 to"
            " k=--num-comps or just k=--num-comps. Disabled by default."
        ),
    )
    parser.add_argument(
        "-c",
        "--cov-type",
        type=str,
        choices=["full", "tied", "diag", "spherical"],
        default="diag",
        help=(
            "The type of covariance to use for the parameterization of"
            " Gaussians. Defaults to diag."
        ),
    )
    parser.add_argument(
        "-i",
        "--init-strat",
        type=str,
        choices=["random", "kmeans", "kmeans++"],
        default="kmeans++",
        help=(
            "Strategy for initializing the parameters of a Gaussian mixture"
            " model. Defaults to kmeans++."
        ),
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1024,
        help="The mini-batch size to use for training. Defaults to 1024",
    )
    parser.add_argument(
        "-e",
        "--max-epochs",
        type=int,
        default=1_000,
        help="Maximum training iterations. Defaults to 1000.",
    )
    parser.add_argument(
        "-r",
        "--cov-regul",
        type=float,
        default=0.01,
        help=(
            "Regularization term for the covariance matrices during training."
            " Defaults to 0.01"
        ),
    )
    parser.add_argument(
        "-d",
        "--data-variant",
        type=str,
        choices=["full-sift", "full-rgb", "red-sift", "red-rgb"],
        default="full-sift",
        help=(
            "When this argument is provided, one of the internal datasets"
            " is used. Does not take precedence over --data-path. Defaults to"
            " full-sift"
        ),
    )
    parser.add_argument(
        "-p",
        "--data-path",
        type=str,
        help=(
            "A valid file path to a user made NumPy file. Takes precedence"
            " over --data-variant."
        ),
    )
    parser.add_argument(
        "-m",
        "--matlab-session",
        type=str,
        help=(
            "Instead of opening a new MATLAB session through matlab.engine,"
            " connect to an existing and shared one."
        ),
    )
    parser.add_argument(
        "--gmm-plot-range",
        type=int,
        default=-1,
        help=(
            "Specifies how many AIC/BIC before and after the minimum ones are"
            " plotted. Value of 0 means plotting only the minimum. Negative"
            " values turns off plotting. Defaults to -1."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="out",
        help=(
            "Specifies the name of the output directory, created in the current"
            " working directory. Specify ONLY the directory name and not a path"
            " to the directory, e.g., data, and NOT /path/to/data. Defaults to"
            " out."
        ),
    )

    return parser.parse_known_args()[0]
