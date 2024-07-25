"""Test the Kalman filter implementation."""

import pytest_cases
import jax.numpy as jnp
import jax

from fpx import fpx


def case_impl_covariance_based():
    return fpx.impl_covariance_based()


def case_impl_cholesky_based():
    return fpx.impl_cholesky_based()


@pytest_cases.parametrize_with_cases("impl", cases=".")
def test_filter_estimates_trajectory_accurately(impl):
    # Set up a test problem
    ts = jnp.linspace(0, 1)
    ssm = fpx.ssm_car_tracking_velocity(ts, noise=1e-4, diffusion=1.0, impl=impl)

    # Create some data
    latent, data = _sample(ssm=ssm, impl=impl)
    assert latent.shape == (len(ts) - 1, 4)
    assert data.shape == (len(ts) - 1, 2)

    # Run a Kalman filter
    estimate = fpx.estimate_filter(impl=impl)
    (mean, cov), _aux = estimate(data, ssm)

    # Assert that the error's magnitude is of the same order
    # as the observation noise
    assert rmse(mean[:2], latent[-1, :2]) < 1e-3


@pytest_cases.parametrize_with_cases("impl", cases=".")
def test_smoother_more_accurate_than_filter(impl):
    # Set up a test problem
    ts = jnp.linspace(0, 1, num=100)
    ssm = fpx.ssm_car_tracking_velocity(ts, noise=1e-1, diffusion=1.0, impl=impl)

    # Create some data
    latent, data = _sample(ssm=ssm, impl=impl)
    assert latent.shape == (len(ts) - 1, 4)
    assert data.shape == (len(ts) - 1, 2)

    # Run a Kalman filter
    estimate = fpx.estimate_filter(impl=impl)
    terminal_filter, _aux = estimate(data, ssm)

    # Run an RTS smoother
    estimate = fpx.estimate_fixedinterval(impl=impl)
    (terminal, conds), aux = estimate(data, ssm)

    marginalize = fpx.sequence_marginalize(impl=impl, reverse=True)
    marginals = marginalize(terminal, conds)

    # Assert that the final states of filter and smoother coincide
    t1 = impl.rv_to_mvnorm(terminal)
    t2 = impl.rv_to_mvnorm(terminal_filter)
    assert jax.tree.all(jax.tree.map(jnp.allclose, t1, t2))

    # Assert that the marginals make sense
    # Select the intermediate states of data, filter, and smoother
    # so that the RMSEs can be compared. Precisely:
    # The smoother-solution does not include the terminal value
    # The filter-solution does not include the initial value
    # The data vectors match the filter solution
    m_f = aux["filter_distributions"].mean[:-1]
    y = latent[:-1]
    m_s = marginals.mean[1:]

    # Assert the filter is better than the noise
    # Use '0.9' to increase the significance
    assert rmse(m_f[:, :2], y[:, :2]) < 0.9 * rmse(data[:-1], y[:, :2])

    # Assert that the smoother is better than the filter
    # Use '0.9' to increase the significance
    assert rmse(m_s, y) < 0.9 * rmse(m_f, y)


@pytest_cases.parametrize_with_cases("impl", cases=".")
def test_state_augmented_filter_matches_rts_smoother_at_initial_state(impl):
    # Set up a test problem
    ts = jnp.linspace(0, 1, num=100)
    ssm = fpx.ssm_car_tracking_velocity(ts, noise=1e-4, diffusion=1.0, impl=impl)

    # Create some data
    latent, data = _sample(ssm=ssm, impl=impl)
    assert latent.shape == (len(ts) - 1, 4)
    assert data.shape == (len(ts) - 1, 2)

    # Run a fixedpoint-smoother via state-augmented filtering
    # and via marginalising over an RTS solution
    estimate = fpx.estimate_fixedpoint_via_fixedinterval(impl=impl)
    initial_rts, _aux = estimate(data, ssm)
    estimate = fpx.estimate_fixedpoint_via_filter(impl=impl)
    initial_fps, _aux = estimate(data, ssm)

    # Check that all leaves match
    initial_rts = impl.rv_to_mvnorm(initial_rts)
    initial_fps = impl.rv_to_mvnorm(initial_fps)
    for x1, x2 in zip(jax.tree.leaves(initial_fps), jax.tree.leaves(initial_rts)):
        assert jnp.allclose(x1, x2, atol=1e-4)


@pytest_cases.parametrize_with_cases("impl", cases=".")
def test_fixedpoint_smoother_matches_state_augmented_filter(impl):
    # Set up a test problem
    ts = jnp.linspace(0, 1, num=100)
    ssm = fpx.ssm_car_tracking_velocity(ts, noise=1e-4, diffusion=1.0, impl=impl)

    # Create some data
    latent, data = _sample(ssm=ssm, impl=impl)
    assert latent.shape == (len(ts) - 1, 4)
    assert data.shape == (len(ts) - 1, 2)

    # Run a fixedpoint-smoother via state-augmented filtering
    # and via marginalising over an RTS solution
    estimate = fpx.estimate_fixedpoint_via_filter(impl=impl)
    initial_rts, _aux = estimate(data, ssm)
    estimate = fpx.estimate_fixedpoint(impl=impl)
    initial_fps, _aux = estimate(data, ssm)

    # Check that all leaves match
    initial_rts = impl.rv_to_mvnorm(initial_rts)
    initial_fps = impl.rv_to_mvnorm(initial_fps)
    for x1, x2 in zip(jax.tree.leaves(initial_fps), jax.tree.leaves(initial_rts)):
        assert jnp.allclose(x1, x2, atol=1e-4)


def test_square_root_parametrisation_matches_conventional_parametrisation_for_filter():
    impl_conv = fpx.impl_covariance_based()
    ts = jnp.linspace(0, 1, num=100)
    ssm_conv = fpx.ssm_car_tracking_velocity(
        ts, noise=1e-4, diffusion=1.0, impl=impl_conv
    )

    # Sample using the conventional parametrisation
    latent, data = _sample(ssm=ssm_conv, impl=impl_conv)
    assert latent.shape == (len(ts) - 1, 4)
    assert data.shape == (len(ts) - 1, 2)

    # Replicate with sqrt parametrisation
    impl_sqrt = fpx.impl_cholesky_based()
    ssm_sqrt = fpx.ssm_car_tracking_velocity(
        ts, noise=1e-4, diffusion=1.0, impl=impl_sqrt
    )

    estimate_conv = fpx.estimate_filter(impl=impl_conv)
    estimate_sqrt = fpx.estimate_filter(impl=impl_sqrt)

    rv_conv, _aux = estimate_conv(data, ssm_conv)
    rv_sqrt, _aux = estimate_sqrt(data, ssm_sqrt)

    rv_conv = impl_conv.rv_to_mvnorm(rv_conv)
    rv_sqrt = impl_sqrt.rv_to_mvnorm(rv_sqrt)

    for x1, x2 in zip(jax.tree.leaves(rv_conv), jax.tree.leaves(rv_sqrt)):
        assert jnp.allclose(x1, x2)


# todo: use our own allclose which depends on the floating-point accuracy?


def _sample(*, ssm, impl):
    key = jax.random.PRNGKey(seed=1)
    key, subkey = jax.random.split(key, num=2)
    x0 = impl.rv_sample(subkey, ssm.init)
    sample = fpx.sequence_sample(impl=impl)
    _, (latent, data) = sample(key, x0, ssm.dynamics)
    return latent, data


def rmse(a, b):
    return jnp.linalg.norm(a - b) / jnp.sqrt(a.size)
