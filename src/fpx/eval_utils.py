"""Evaluation utilities."""

import pickle

import jax.numpy as jnp
import pandas as pd

from fpx import eval_utils


def filename_results(file: str, /, *, replace: str):
    """Create a filename: results/<matching-directory>/results.pkl."""
    dirname = str(file).replace(replace, "results")
    dirname = dirname.replace(".py", "")
    return f"{dirname}/results.pkl"


def format_large_number_tex(float_number):
    """Format a large number to tex-compatible scientific notation."""
    # Taken from:
    # https://stackoverflow.com/questions/41157879/python-pandas-how-to-format-big-numbers-in-powers-of-ten-in-latex
    exponent = jnp.floor(jnp.log10(float_number))
    mantissa = float_number / 10**exponent
    mantissa_format = str(mantissa)[0:3]  # todo: expose num_digits?
    return r"${0} \times 10^{{{1}}}$".format(mantissa_format, str(int(exponent)))
