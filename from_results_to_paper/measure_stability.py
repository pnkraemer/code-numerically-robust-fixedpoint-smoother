"""Turn the stability-results into something to include in the paper."""

import os
import pickle

import jax.numpy as jnp
import pandas as pd
from fpx import eval_utils

# Load the results
dirname = eval_utils.matching_directory(__file__, replace="from_results_to_paper")
filename = f"{dirname}/results.pkl"
with open(filename, "rb") as f:
    results = pickle.load(f)

# Turn the dictionary into a dataframe
df = pd.DataFrame(results)

# Format large numbers
df_formatted = df.map(lambda s: eval_utils.format_large_number_tex(4 * s))

# Print the dataframe in a latex-compatible way (to be copy/pasted)
print()
print()
print(df_formatted.to_latex())
print()
print()
print(df_formatted.T.to_latex())
print()
print()
