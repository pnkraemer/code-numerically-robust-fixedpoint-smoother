import jax.numpy as jnp
import jax
import time

from fpx import fpx


def main(seed: int, implementation: fpx.Impl, /, fixedpoint, nruns, ndim, nsteps):
    # Set up a test problem
    ts = jnp.linspace(0, 1, num=nsteps)
    ssm = fpx.ssm_car_tracking_acceleration(
        ts, noise=1e-4, diffusion=1.0, impl=impl, dim=ndim
    )

    # Create some data
    key = jax.random.PRNGKey(seed=seed)
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
    seed = 3
    num_runs = 3
    num_dims = 5
    num_steps = 100

    print()
    print(f"\nSquare-root code (fastest of n={num_runs} runs):")
    print("-----------------------------------------------------")
    impl = fpx.impl_square_root()
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

    print(f"\nConventional code (fastest of n={num_runs} runs):")
    print("-----------------------------------------------------")
    impl = fpx.impl_conventional()
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
