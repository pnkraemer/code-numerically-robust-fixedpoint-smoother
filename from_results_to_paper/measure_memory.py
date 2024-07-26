import pickle

import jax.numpy as jnp
import pandas as pd


def format_tex(float_number):
    # Taken from:
    # https://stackoverflow.com/questions/41157879/python-pandas-how-to-format-big-numbers-in-powers-of-ten-in-latex
    exponent = jnp.floor(jnp.log10(float_number))
    mantissa = float_number / 10**exponent
    mantissa_format = str(mantissa)[0:3]
    return r"${0} \times 10^{{{1}}}$".format(mantissa_format, str(int(exponent)))


def sizeof_fmt(num, suffix="B"):
    # https://stackoverflow.com/questions/1094841/get-a-human-readable-version-of-a-file-size
    for unit in ("", "K", "M", "G", "T", "P", "E", "Z"):
        if abs(num) < 1024.0:
            return f"{num:3.0f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


dirname = str(__file__)
dirname = dirname.replace("from_results_to_paper", "results")
dirname = dirname.replace(".py", "")

filename = f"{dirname}/results.pkl"
with open(filename, "rb") as f:
    results = pickle.load(f)


for label, value in results.items():
    print(label)
    print(value)

    df = pd.DataFrame(value)

    df2 = df.map(lambda s: format_tex(4 * s))
    # df2 = df.map(lambda s: sizeof_fmt(4 * s))
    print()

    print()
    print(df.to_latex())
    print()
    print()
    print(df2.to_latex())
    print()
    print()
