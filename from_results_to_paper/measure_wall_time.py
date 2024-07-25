import pickle
import pandas as pd

dirname = str(__file__)
dirname = dirname.replace("from_results_to_paper", "results")
dirname = dirname.replace(".py", "")

filename = f"{dirname}/results.pkl"
with open(filename, "rb") as f:
    results = pickle.load(f)

df = pd.DataFrame(results)
print()
print()
print()
print()
print(df.to_latex())
print()
print()
print()
print()
print(df.T.to_latex())
print()
print()
print()
print()
