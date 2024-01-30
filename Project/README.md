# PyCave GMM with AIC and BIC computation

PyGMM is a Python package that trains one or multiple GMMs with GPU acceleration
and outputs the results in a MATLAB file, through the official MATLAB to Python
interface. The requirements and setup instructions are written below.

Run `python -m pygmm -h` for help with the flags and parameterization that can
be done.

## Setup

Requires python >= 3.8 and < 3.11 .

Before running, a virtual env (venv) must be created in the current folder and
the appropriate dependencies, as described below.

Firstly, open a terminal and change to the directory where this repository is
cloned. Then, run the following command

```bash
python -m venv ./
```

Once it is done, activate the venv with the following script, based on your OS
and terminal combination.

For Windows using the standard command prompt:

```bash
.\Scripts\activate.bat
```

For PowerShell terminal:

```bash
.\Scripts\Activate.ps1
```

For bash/zsh terminals:

```bash
.\Scripts\activate
```

Once the venv is activated, run the following commands, in the order listed, to
install the dependencies:

```bash
# Installs the dependencies (including PyTorch without GPU support)
# from the default index
python -m pip install -r ./requirements.txt

# Installs PyTorch from PyTorch's index, which has GPU support.
python -m pip install --force-reinstall -r ./torch-req.txt
```

Lastly, install MATLAB Engine for Python by following the [official instructions](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).

Once these are done, run the [pygmm](./pygmm/__main__.py) script and wait for
the results.

```bash
python -m pygmm
```

## License

[MIT License](./LICENSE).
