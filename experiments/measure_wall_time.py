"""Measure the wall time of fixedpoint estimation via differnet methods."""

import argparse
import os
import pickle
import time

import jax
import jax.numpy as jnp
import tqdm
from fpx import eval_utils, fpx


def main(*, num_runs=3, seed=1, num_steps=1_000, save=True) -> None:
    # Max dimension is 100, because for e.g. 128, I run out of RAM...
    num_dims_list = [2, 5, 10, 20, 50, 100]

    # Run the experiment
    key = jax.random.PRNGKey(seed)
    results = run_experiment(
        num_runs=num_runs,
        key=key,
        num_dims_list=num_dims_list,
        num_steps=num_steps,
    )

    # Save results to a file
    if save:
        dirname = eval_utils.matching_directory(__file__, replace="experiments")
        os.makedirs(dirname, exist_ok=True)
        with open(f"{dirname}/results.pkl", "wb") as f:
            pickle.dump(results, f)

        print(f"\nSaved results to {dirname}/results.pkl.\n")


def run_experiment(*, key, num_runs, num_dims_list, num_steps) -> dict:
    # It's all Cholesky-based for this experiment
    impl = fpx.impl_cholesky_based()

    results: dict = {}
    progressbar = tqdm.tqdm(num_dims_list)
    for num_dims in progressbar:
        progressbar.set_description(f"d={num_dims}")

        label_num_dims = f"$d={num_dims}$"
        results[label_num_dims] = {}

        # Set up a state-space model
        ts = jnp.linspace(0, 1, num=num_steps)
        ssm = fpx.ssm_regression_wiener_velocity(ts, impl=impl, dim=num_dims)

        # Randomly sample values of a state-space model. The values don't matter.
        key, subkey = jax.random.split(key, num=2)
        ssm = eval_utils.tree_random_like(subkey, ssm, scale=1.0 / num_steps)

        # Sample data
        key, subkey = jax.random.split(key, num=2)
        sample = fpx.compute_stats_sample(impl=impl)
        sample = jax.jit(sample)
        (_latent, data) = sample(key, ssm)
        assert not jnp.any(jnp.isnan(data))

        # Compute the wall-time of all three methods
        via_filter = fpx.compute_fixedpoint_via_filter(impl=impl)
        via_fixedinterval = fpx.compute_fixedpoint_via_smoother(impl=impl)
        via_recursion = fpx.compute_fixedpoint(impl=impl)
        for name, estimate in [
            ("Via recursion", via_recursion),
            ("Via fixed-interval", via_fixedinterval),
            ("Via filter", via_filter),
        ]:
            estimate = jax.jit(estimate)
            t = measure_wall_time(estimate, ssm=ssm, data=data, num_runs=num_runs)
            results[label_num_dims][name] = min(t)
    return results


def measure_wall_time(fixedpoint, *, data, ssm, num_runs) -> list[float]:
    """Measure the wall-time of a fixed-point smoother."""

    # Recompile (better safe than sorry...)
    fixedpoint = jax.jit(fixedpoint)

    # Execute once to pre-compile (and to compute errors)
    initial_rts, _aux = fixedpoint(data, ssm)
    if jnp.any(jnp.isnan(initial_rts.mean)):
        print("NaN detected")
        return [-1.0]

    ts = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        initial_rts, _ = fixedpoint(data, ssm)
        initial_rts[0].block_until_ready()
        initial_rts[1].block_until_ready()
        t1 = time.perf_counter()
        ts.append(t1 - t0)
    return ts


if __name__ == "__main__":
    main()
