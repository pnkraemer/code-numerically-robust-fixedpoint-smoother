"""Turn the robustness-results into something to include in the paper."""

import os
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
from fpx import eval_utils

# Load the results
dirname = eval_utils.matching_directory(__file__, replace="from_results_to_paper")
filename = f"{dirname}/results.pkl"
with open(filename, "rb") as f:
    results = pickle.load(f)

ts = jnp.load(f"{dirname}/ts.npy")
means = jnp.load(f"{dirname}/mean.npy")

# Plot the solution
plt.rcParams.update(eval_utils.plot_style())
plt.subplots(figsize=(2.5, 2), tight_layout=True)
plt.plot(ts, means[:, 0])
plt.plot(
    ts[0],
    means[0, 0],
    marker=".",
    markerfacecolor="white",
    linestyle="None",
    color="C0",
)
plt.plot(
    ts[-1],
    means[-1, 0],
    marker=".",
    markerfacecolor="white",
    linestyle="None",
    color="C0",
)
# plt.title(r"Problem $\#$15", fontsize="medium")
plt.xlabel("Input $t$")
plt.ylabel("Output $x$")

# Save the plot
name = str(__file__)
name = name.replace(".py", "")
plt.savefig(f"{name}.pdf")
plt.show()

# Turn the dictionary into a dataframe
df = pd.DataFrame(results)

# Format large numbers
df_formatted = df.map(lambda s: eval_utils.format_large_number_tex(s, num_digits=1))

# Print the dataframe in a latex-compatible way (to be copy/pasted)
print()
print()
print(df_formatted.to_latex())
print()
print()
