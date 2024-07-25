"""Measure the wall time of fixedpoint estimation via differnet methods."""

import jax.numpy as jnp
import jax
import time

from fpx import fpx
import argparse


# todo: save results to a dictionary
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--num_dims", type=int, default=2)
    parser.add_argument("--num_steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    print(args)

    key = jax.random.PRNGKey(args.seed)

    for impl in [fpx.impl_covariance_based(), fpx.impl_cholesky_based()]:
        print()
        print(impl.name)
        print(f"{'-'*len(impl.name)}")

        # Set up a test problem
        ts = jnp.linspace(0, 1, num=args.num_steps)
        ssm = fpx.ssm_car_tracking_acceleration(
            ts, noise=1e-4, diffusion=1.0, impl=impl, dim=args.num_dims
        )
        sample = fpx.compute_stats_sample(impl=impl)
        data = sample_data(key, ssm=ssm, sample=sample)

        # Compute a reference solution
        estimate = jax.jit(fpx.compute_fixedpoint(impl=impl))
        ref, _ = estimate(data, ssm)

        for name, estimate in [
            ("filter", fpx.compute_fixedpoint_via_filter(impl=impl)),
            ("fixed-interval", fpx.compute_fixedpoint_via_fixedinterval(impl=impl)),
            ("recursion", fpx.compute_fixedpoint(impl=impl)),
        ]:
            print(f"\nFixedpoint via {name}")
            estimate = jax.jit(estimate)
            t = benchmark(estimate, ref=ref, ssm=ssm, data=data, num_runs=args.num_runs)
            print(f"\t {min(t):.1e}")


def benchmark(fixedpoint, *, ref, data, ssm, num_runs):
    # Execute once to pre-compile (and to compute errors)
    initial_rts, _ = fixedpoint(data, ssm)
    if jnp.any(jnp.isnan(initial_rts.mean)):
        print("NaN detected")
        return [-1.0]

    # If the values don't match, abort
    if not jnp.allclose(initial_rts.mean, ref.mean, atol=1e-3, rtol=1e-3):
        print("Values do not match the reference")
        print(f"{jnp.abs(initial_rts.mean - ref.mean)}")
        return [-1.0]

    ts = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        initial_rts, _ = fixedpoint(data, ssm)
        initial_rts.mean.block_until_ready()
        t1 = time.perf_counter()
        ts.append(t1 - t0)
    return ts


def sample_data(key, *, ssm, sample):
    (latent, data) = sample(key, ssm)
    return data


if __name__ == "__main__":
    main()
