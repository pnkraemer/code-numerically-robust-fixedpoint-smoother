"""Measure the wall time of fixedpoint estimation via differnet methods."""

import jax.numpy as jnp
import jax
import time

from fpx import fpx
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", default=5)
    parser.add_argument("--num_dims", default=2)
    parser.add_argument("--num_steps", default=1_000)
    parser.add_argument("--seed", default=1)
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

        print("\nFixedpoint via filter")
        estimate = jax.jit(fpx.compute_fixedpoint_via_filter(impl=impl))
        t = benchmark(estimate, ssm=ssm, data=data, num_runs=args.num_runs)
        print(f"\t {min(t):.1e}")

        print("\nFixedpoint via fixed-interval")
        estimate = jax.jit(fpx.compute_fixedpoint_via_fixedinterval(impl=impl))
        t = benchmark(estimate, ssm=ssm, data=data, num_runs=args.num_runs)
        print(f"\t {min(t):.1e}")

        print("\nFixedpoint via recursion")
        estimate = jax.jit(fpx.compute_fixedpoint(impl=impl))
        t = benchmark(estimate, ssm=ssm, data=data, num_runs=args.num_runs)
        print(f"\t {min(t):.1e}")

        print()


def benchmark(fixedpoint, *, data, ssm, num_runs):
    # Create some data

    # Run a fixedpoint-smoother via state-augmented filtering
    # and via marginalising over an RTS solution
    # Execute once to pre-compile
    initial_rts, _ = fixedpoint(data, ssm)
    if jnp.any(jnp.isnan(initial_rts.mean)):
        print("NaN detected")
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
