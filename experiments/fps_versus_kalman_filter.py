import jax.numpy as jnp
import jax
import time

from fpx import fpx


# todo: count the number of NaN runs for different seeds and for increasing Ns
# todo: test the time of all methods for increasing N and increasing d
#  (currently, this script does it all, but nothing well)
# get inspired by Yaghoobi et al. 2022 (parallel sigma point..., reference is in the paper draft)


def main(seed_: int, implementation: fpx.Impl, /, fixedpoint, nruns, ndim, nsteps):
    # Set up a test problem
    ts = jnp.linspace(0, 1, num=nsteps)
    ssm = fpx.ssm_car_tracking_acceleration(
        ts, noise=1e-4, diffusion=1.0, impl=impl, dim=ndim
    )

    # Create some data
    key = jax.random.PRNGKey(seed=seed_)
    key, subkey = jax.random.split(key, num=2)
    x0 = impl.rv_sample(subkey, ssm.init)
    _, (latent, data) = fpx.sequence_sample(key, x0, ssm.dynamics, impl=implementation)

    # Run a fixedpoint-smoother via state-augmented filtering
    # and via marginalising over an RTS solution
    # Execute once to pre-compile
    initial_rts, _ = fixedpoint(data, ssm, impl=implementation)
    if jnp.any(jnp.isnan(initial_rts.mean)):
        print("NaN detected")
        return [-1.0]

    ts = []
    for _ in range(nruns):
        t0 = time.perf_counter()
        initial_rts, _ = fixedpoint(data, ssm, impl=implementation)
        initial_rts.mean.block_until_ready()
        t1 = time.perf_counter()
        ts.append(t1 - t0)
    return ts


if __name__ == "__main__":
    seed = 3  # todo: check 100 seeds and count the number of NaN runs for stability
    num_runs = 3
    num_dims = 2
    num_steps = 1000  # todo: increase num_steps to plot stability and costs

    for seed in range(10):
        for label, impl in [
            ("Covariance-based", fpx.impl_covariance_based()),
            ("Cholesky-based", fpx.impl_cholesky_based()),
        ]:
            print()
            print(f"\n{label} code (fastest of n={num_runs} runs):")
            print("-----------------------------------------------------")
            print("Via filter:")
            t = main(
                seed,
                impl,
                fixedpoint=fpx.estimate_fixedpoint_via_filter,
                nruns=num_runs,
                ndim=num_dims,
                nsteps=num_steps,
            )
            print("\t", min(t))
            print("Via fixed-interval smoother:")
            t = main(
                seed,
                impl,
                fixedpoint=fpx.estimate_fixedpoint_via_fixedinterval,
                nruns=num_runs,
                ndim=num_dims,
                nsteps=num_steps,
            )
            print("\t", min(t))
            print("Via proper recursion:")
            t = main(
                seed,
                impl,
                fixedpoint=fpx.estimate_fixedpoint,
                nruns=num_runs,
                ndim=num_dims,
                nsteps=num_steps,
            )
            print("\t", min(t))
            print()
