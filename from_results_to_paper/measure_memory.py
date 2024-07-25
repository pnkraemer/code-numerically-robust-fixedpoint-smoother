import pickle
import pandas as pd
import jax

dirname = str(__file__)
dirname = dirname.replace("from_results_to_paper", "results")
dirname = dirname.replace(".py", "")

filename = f"{dirname}/results.pkl"
with open(filename, "rb") as f:
    results = pickle.load(f)

results = jax.tree.map(int, results)

for label, value in results.items():
    print(label)
    print(value)

    df = pd.DataFrame(value)
    df = df.style.format(decimal=".", thousands=",")
    print()
    print()
    print(df.to_latex())
    print()
    print()
    print(df.T.to_latex())
    print()
    print()
