# code-numerically-stable-fixed-point-smoother

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

- [ ] Square-root fixed-point smoother versus a state-augmented square-root Kalman filter
- [ ] Square-root fixed-point smoother versus non-square-root fixed-point smoother
- [ ] Parameter-estimation in boundary value problems via probabilistic numerics

## Using the code

Everything is contained in a single module.
To use it, and after installation, import
```python
from fpx import fpx

print(help(fpx))
```
and access all code via `fpx.*` ("fpx" stands for "fixed-point smoothing in JAX").
Consult the test file in `tests/test_fpx.py` for examples.

## Working with the source

After following the installation instructions above, the test-dependencies are installed.
To run the tests, run
```commandline
make test
```
