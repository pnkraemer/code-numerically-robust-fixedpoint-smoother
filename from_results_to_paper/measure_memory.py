import pickle

import jax.numpy as jnp
import pandas as pd
from fpx import eval_utils

# Load the results
dirname = eval_utils.matching_directory(__file__, replace="from_results_to_paper")
filename = f"{dirname}/results.pkl"
with open(filename, "rb") as f:
    results = pickle.load(f)

# Turn the dict into a data-frame
df = pd.DataFrame(results)

# Format the numbers
# Multiply the strings by 4 because this turns
# the number of single-precision floats into bytes
# (32-bit precision; 8 bits are one byte)
df_formatted = df.map(lambda s: eval_utils.format_large_number_tex(4 * s))

# Print a latex-compatible version of the frame
print()
print()
print(df_formatted.to_latex())
print()
print()
print(df_formatted.T.to_latex())
print()
print()
