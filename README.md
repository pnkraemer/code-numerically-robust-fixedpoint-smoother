# code-numerically-robust-fixed-point-smoother

**Warning:**
This is experiment code.
But if you want to work with this repository, proceed as follows.


## Installation

We use Python 3.10 for all experiments.
Older versions might also work.

First, ensure that JAX is installed.
Then, run
```commandline
pip install .
```
which installs the source code plus all dependencies.

## Experiments

- [x] Wall-time and memory on an autoregressive SSM
- [x] Numerical stability for solving a BVP
- [x] Parameter estimation: Estimate the initial location of a moving object (eg, a car)


## Using the code

Everything is contained in a single module.
To use it, and after installation, import
```python
from fpx import fpx

print(help(fpx))
```
and access all code via `fpx.*` ("fpx" stands for "fixed-point smoothing in JAX").
Consult the test file in `tests/test_fpx.py` for examples.

You may also run `mkdocs serve` to get a list of all types and functions.

## Working with the source

After following the installation instructions above, the test-dependencies are installed.
To run the tests, run
```commandline
make test
```
