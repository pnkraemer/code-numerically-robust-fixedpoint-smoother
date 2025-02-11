# code-numerically-robust-fixed-point-smoother



This repository contains the code for the paper

> Krämer, Nicholas. 
> "Numerically Robust Fixed-Point Smoothing Without State Augmentation." 
> Transactions on Machine Learning Research (2025).

Here is a bibtex entry:

```bibtex
@article{
  kramer2025numerically,
  title={Numerically Robust Fixed-Point Smoothing Without State Augmentation},
  author={Nicholas Kr{\"a}mer},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=LVQ8BEL5n3},
  note={}
}```


## Warning 
This is experiment code.
But if you want to work with this repository, proceed as follows.


## Installation

We use Python 3.10 for all experiments.
Other versions might also work.

First, ensure that JAX is installed.
Then, run
```commandline
pip install .
```
which installs the source code plus all dependencies.

## Experiments

To run the experiments, execute (for instance)
```commandline
python experiments/estimate_parameters.py
```
or run all experiments via
```commandline
make run-experiments
```
To turn the results into the tables from the Paper, execute the scripts in `from_results_to_paper/*`.
The scripts' names match the experiments' names, for example,
```commandline
python from_results_to_paper/measure_robustness.py
```
The parameter estimation experiment plots result in the experiment script.

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
