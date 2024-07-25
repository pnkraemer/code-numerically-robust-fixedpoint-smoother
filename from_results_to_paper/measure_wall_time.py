import pickle
import pandas as pd

filename = str(__file__)
filename = filename.replace("from_results_to_paper", "results")
filename = filename.replace(".py", "_n1000_d2.pkl")

with open(filename, "rb") as f:
    results = pickle.load(f)

df = pd.DataFrame(results)
print(df)
