
import pytest_cases
import jax.numpy as jnp
import jax
import time

from fpx import fpx

def main():
    # It's all square-root implementation
    impl = fpx.impl_square_root()

    # Set up a test problem
    ts = jnp.linspace(0, 1, num=100)
    ssm = fpx.ssm_car_tracking_velocity(ts, noise=1e-4, diffusion=1.0, impl=impl, dim=50)

    # Create some data
    key = jax.random.PRNGKey(seed=1)
    key, subkey = jax.random.split(key, num=2)
    x0 = impl.rv_sample(subkey, ssm.init)
    _, (latent, data) = fpx.sequence_sample(key, x0, ssm.dynamics, impl=impl)


    # Run a fixedpoint-smoother via state-augmented filtering
    # and via marginalising over an RTS solution
    # Execute once to pre-compile
    initial_rts, _ = fpx.estimate_fixedpoint_via_filter(data, ssm, impl=impl)
    initial_fps, _ = fpx.estimate_fixedpoint(data, ssm, impl=impl)

    t0 = time.perf_counter()
    for _ in range(10):
        initial_rts, _ = fpx.estimate_fixedpoint_via_filter(data, ssm, impl=impl)
        initial_rts.mean.block_until_ready()
    t1 = time.perf_counter()
    print("Via filter:", t1 - t0)

    t0 = time.perf_counter()
    for _ in range(10):
        initial_rts, _ = fpx.estimate_fixedpoint(data, ssm, impl=impl)
        initial_rts.mean.block_until_ready()
    t1 = time.perf_counter()
    print("Via FPS:", t1 - t0)

    # Check that all leaves match
    initial_rts = impl.rv_to_mvnorm(initial_rts)
    initial_fps = impl.rv_to_mvnorm(initial_fps)
    for x1, x2 in zip(jax.tree.leaves(initial_fps), jax.tree.leaves(initial_rts)):
        assert jnp.allclose(x1, x2, atol=1e-4)

if __name__ == "__main__":
    main()