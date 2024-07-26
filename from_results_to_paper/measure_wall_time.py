"""Display the wall-time results in a paper-compatible format."""

import pickle

import jax.numpy as jnp
import pandas as pd
from fpx import eval_utils

# Load the results
filename = eval_utils.filename_results(__file__, replace="from_results_to_paper")
with open(filename, "rb") as f:
    results = pickle.load(f)

df = pd.DataFrame(results)
df = df.map(lambda x: eval_utils.format_large_number_tex(x))
print()
print()
print(df.to_latex())
print()
print()
print(df.T.to_latex())
print()
print()
