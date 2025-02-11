"""Measure the memory consumption of different fixed-point smoothers."""

import argparse
import os
import pickle

import jax
import jax.numpy as jnp
import tqdm
from fpx import eval_utils, fpx


def main(*, seed=1, num_steps=1_000, save=True):
    """Run the experiment."""
    key = jax.random.PRNGKey(seed)
    num_dims_list = [2, 5, 10, 20, 50, 100]
    results = measure_memory(key=key, num_steps=num_steps, num_dims_list=num_dims_list)

    if save:
        dirname = eval_utils.matching_directory(__file__, replace="experiments")
        os.makedirs(dirname, exist_ok=True)
        with open(f"{dirname}/results.pkl", "wb") as f:
            pickle.dump(results, f)

        print(f"\nSaved results to {dirname}/results.pkl.\n")


def measure_memory(*, key, num_steps, num_dims_list) -> dict:
    """Measure the memory consumption of fixed-point smoothers."""
    # It's all Cholesky-based for this experiment
    impl = fpx.impl_cholesky_based()

    results: dict = {}
    progressbar = tqdm.tqdm(num_dims_list)
    for num_dims in progressbar:
        progressbar.set_description(f"d={num_dims}")

        # Turn the num_dims into a label
        label_num_dims = f"$d={num_dims}$"
        results[label_num_dims] = {}

        # Set up a state-space model
        ts = jnp.linspace(0, 1, num=num_steps)
        ssm_fun = fpx.ssm_regression_wiener_velocity(ts, impl=impl, dim=num_dims)
        ssm = ssm_fun(noise=1.0, diffusion=1.0)  # values don't matter

        # Randomly sample values of a state-space model. The values don't matter.
        key, subkey = jax.random.split(key, num=2)
        ssm = eval_utils.tree_random_like(subkey, ssm, scale=1.0 / num_steps)

        # Sample data
        key, subkey = jax.random.split(key, num=2)
        sample = fpx.compute_stats_sample(impl=impl)
        sample = jax.jit(sample)
        (_latent, data) = sample(key, ssm)
        assert not jnp.any(jnp.isnan(data))

        # Measure the size of the state via a callback
        def callback(*x):
            return {"size": jax.flatten_util.ravel_pytree(x)[0].size}

        # Run all methods with the callback
        via_filter = fpx.compute_fixedpoint_via_filter(impl=impl, cb=callback)
        via_fixedinterval = fpx.compute_fixedpoint_via_smoother(impl=impl, cb=callback)
        via_recursion = fpx.compute_fixedpoint(impl=impl, cb=callback)
        for name, estimate in [
            ("Via recursion", via_recursion),
            ("Via filter", via_filter),
            ("Via fixed-interval", via_fixedinterval),
        ]:
            _, aux = estimate(data, ssm=ssm)
            results[label_num_dims][name] = jnp.sum(aux["size"])

    return results


if __name__ == "__main__":
    main()
