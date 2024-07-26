"""Evaluation utilities."""

import pickle

import jax.numpy as jnp
import pandas as pd

from fpx import eval_utils


def matching_directory(file: str, /, *, replace: str):
    """Create a directory in results/ whose name matches the current filename."""
    dirname = str(file).replace(replace, "results")
    return dirname.replace(".py", "")


def format_large_number_tex(float_number):
    """Format a large number to tex-compatible scientific notation."""
    # Taken from:
    # https://stackoverflow.com/questions/41157879/python-pandas-how-to-format-big-numbers-in-powers-of-ten-in-latex
    exponent = jnp.floor(jnp.log10(float_number))
    mantissa = float_number / 10**exponent
    mantissa_format = str(mantissa)[0:3]  # todo: expose num_digits?
    return r"${0} \times 10^{{{1}}}$".format(mantissa_format, str(int(exponent)))
