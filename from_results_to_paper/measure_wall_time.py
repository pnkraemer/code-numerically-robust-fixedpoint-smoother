import pickle

filename = str(__file__)
filename = filename.replace("from_results_to_paper", "results")
filename = filename.replace(".py", ".pkl")

with open(filename, "rb") as f:
    results = pickle.load(f)

print(results)
