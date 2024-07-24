import jax.numpy as jnp
import jax
import time

from fpx import fpx


def main(impl: fpx.Impl, fixedpoint):
    # Set up a test problem
    ts = jnp.linspace(0, 1, num=100)
    ssm = fpx.ssm_car_tracking_velocity(
        ts, noise=1e-8, diffusion=1.0, impl=impl, dim=50
    )

    # Create some data
    key = jax.random.PRNGKey(seed=1)
    key, subkey = jax.random.split(key, num=2)
    x0 = impl.rv_sample(subkey, ssm.init)
    _, (latent, data) = fpx.sequence_sample(key, x0, ssm.dynamics, impl=impl)

    # Run a fixedpoint-smoother via state-augmented filtering
    # and via marginalising over an RTS solution
    # Execute once to pre-compile
    initial_rts, _ = fixedpoint(data, ssm, impl=impl)
    assert not jnp.any(jnp.isnan(initial_rts.mean))

    t0 = time.perf_counter()
    for _ in range(10):
        initial_rts, _ = fixedpoint(data, ssm, impl=impl)
        initial_rts.mean.block_until_ready()
    t1 = time.perf_counter()
    return t1 - t0


if __name__ == "__main__":
    impl = fpx.impl_conventional()
    t = main(impl=impl, fixedpoint=fpx.estimate_fixedpoint_via_filter)
    print("Conventional via filter:\n\t", t)
    t = main(impl=impl, fixedpoint=fpx.estimate_fixedpoint_via_fixedinterval)
    print("Conventional via fixed-interval smoother:\n\t", t)
    t = main(impl=impl, fixedpoint=fpx.estimate_fixedpoint)
    print("Conventional via proper recursion:\n\t", t)

    impl = fpx.impl_square_root()
    t = main(impl=impl, fixedpoint=fpx.estimate_fixedpoint_via_filter)
    print("Sqrt via filter:\n\t", t)
    t = main(impl=impl, fixedpoint=fpx.estimate_fixedpoint_via_fixedinterval)
    print("Sqrt via fixed-interval smoother:\n\t", t)
    t = main(impl=impl, fixedpoint=fpx.estimate_fixedpoint)
    print("Sqrt via proper recursion:\n\t", t)
