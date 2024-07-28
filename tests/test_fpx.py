"""Test the Kalman filter implementation."""

import jax
import jax.numpy as jnp
import pytest_cases
from fpx import fpx


def case_impl_covariance_based():
    return fpx.impl_covariance_based()


def case_impl_cholesky_based():
    return fpx.impl_cholesky_based()


@pytest_cases.parametrize(
    "compute_fun",
    [
        fpx.compute_fixedpoint,
        fpx.compute_fixedpoint_via_smoother,
        fpx.compute_fixedpoint_via_filter,
        fpx.compute_fixedinterval,
        fpx.compute_filter,
    ],
)
def test_estimators_accept_callbacks(compute_fun):
    # Set up a test problem
    impl = fpx.impl_covariance_based()
    ts = jnp.linspace(0, 1)
    ssm_parametrize = fpx.ssm_regression_wiener_velocity(ts, impl=impl, dim=2)
    ssm = ssm_parametrize(noise=1e-4, diffusion=1.0)

    # Create some data
    latent, data = _sample(ssm=ssm, impl=impl)
    assert latent.shape == (len(ts), 4)
    assert data.shape == (len(ts) - 1, 2)

    # Sanity check: aux is always a dict, and callbacks are optional
    estimate = compute_fun(impl=impl)
    _, aux = estimate(data, ssm)
    assert isinstance(aux, dict)

    # Test: accepts callback and returns info at top-level
    def callback(*args):
        return {"size": jax.flatten_util.ravel_pytree(args)[0].size}

    estimate = compute_fun(impl=impl, cb=callback)
    _, aux = estimate(data, ssm)
    assert "size" in aux.keys()


@pytest_cases.parametrize_with_cases("impl", cases=".")
def test_filter_estimates_trajectory_accurately(impl):
    # Set up a test problem
    ts = jnp.linspace(0, 1)
    ssm_parametrize = fpx.ssm_regression_wiener_velocity(ts, impl=impl, dim=2)
    ssm = ssm_parametrize(noise=1e-4, diffusion=1.0)

    # Create some data
    latent, data = _sample(ssm=ssm, impl=impl)
    assert latent.shape == (len(ts), 4)
    assert data.shape == (len(ts) - 1, 2)

    # Run a Kalman filter
    estimate = fpx.compute_filter(impl=impl)
    (mean, cov), _aux = estimate(data, ssm)

    # Assert that the error's magnitude is of the same order
    # as the observation noise
    assert rmse(mean[:2], latent[-1, :2]) < 1e-3


@pytest_cases.parametrize_with_cases("impl", cases=".")
def test_smoother_more_accurate_than_filter(impl):
    # Set up a test problem
    ts = jnp.linspace(0, 1, num=100)
    ssm_parametrize = fpx.ssm_regression_wiener_velocity(ts, impl=impl, dim=2)
    ssm = ssm_parametrize(noise=1e-1, diffusion=1.0)

    # Create some data
    latent, data = _sample(ssm=ssm, impl=impl)
    assert latent.shape == (len(ts), 4)
    assert data.shape == (len(ts) - 1, 2)

    # Run a Kalman filter
    estimate = fpx.compute_filter(impl=impl)
    terminal_filter, _aux = estimate(data, ssm)

    # Run an RTS smoother
    estimate = fpx.compute_fixedinterval(impl=impl)
    (terminal, conds), aux = estimate(data, ssm)

    marginalize = fpx.compute_stats_marginalize(impl=impl, reverse=True)
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
    m_f = aux["filter_distributions"].mean
    y = latent
    m_s = marginals.mean

    # Assert the filter is better than the noise
    # Use '0.9' to increase the significance
    assert rmse(m_f[:, :2], y[1:, :2]) < 0.9 * rmse(data, y[1:, :2])

    # Assert that the smoother is better than the filter
    # Use '0.9' to increase the significance
    assert rmse(m_s, y) < 0.9 * rmse(m_f, y[1:])


@pytest_cases.parametrize_with_cases("impl", cases=".")
def test_state_augmented_filter_matches_rts_smoother_at_initial_state(impl):
    # Set up a test problem
    ts = jnp.linspace(0, 1, num=100)
    ssm_parametrize = fpx.ssm_regression_wiener_velocity(ts, impl=impl, dim=2)
    ssm = ssm_parametrize(noise=1e-4, diffusion=1.0)

    # Create some data
    latent, data = _sample(ssm=ssm, impl=impl)
    assert latent.shape == (len(ts), 4)
    assert data.shape == (len(ts) - 1, 2)

    # Run a fixedpoint-smoother via state-augmented filtering
    # and via marginalising over an RTS solution
    estimate = fpx.compute_fixedpoint_via_smoother(impl=impl)
    initial_rts, _aux = estimate(data, ssm)
    estimate = fpx.compute_fixedpoint_via_filter(impl=impl)
    initial_fps, _aux = estimate(data, ssm)

    # Check that all leaves match
    initial_rts = impl.rv_to_mvnorm(initial_rts)
    initial_fps = impl.rv_to_mvnorm(initial_fps)
    for x1, x2 in zip(jax.tree.leaves(initial_fps), jax.tree.leaves(initial_rts)):
        assert allclose(x1, x2)


@pytest_cases.parametrize_with_cases("impl", cases=".")
def test_fixedpoint_smoother_matches_state_augmented_filter(impl):
    # Set up a test problem
    ts = jnp.linspace(0, 1, num=100)
    ssm_parametrize = fpx.ssm_regression_wiener_velocity(ts, impl=impl, dim=2)
    ssm = ssm_parametrize(noise=1e-4, diffusion=1.0)

    # Create some data
    latent, data = _sample(ssm=ssm, impl=impl)
    assert latent.shape == (len(ts), 4)
    assert data.shape == (len(ts) - 1, 2)

    # Run a fixedpoint-smoother via state-augmented filtering
    # and via marginalising over an RTS solution
    estimate = fpx.compute_fixedpoint_via_filter(impl=impl)
    initial_rts, _aux = estimate(data, ssm)
    estimate = fpx.compute_fixedpoint(impl=impl)
    initial_fps, _aux = estimate(data, ssm)

    # Check that all leaves match
    initial_rts = impl.rv_to_mvnorm(initial_rts)
    initial_fps = impl.rv_to_mvnorm(initial_fps)
    for x1, x2 in zip(jax.tree.leaves(initial_fps), jax.tree.leaves(initial_rts)):
        assert allclose(x1, x2)


def test_square_root_parametrisation_matches_conventional_parametrisation_for_filter():
    impl_conv = fpx.impl_covariance_based()
    ts = jnp.linspace(0, 1, num=100)
    ssm_parametrize = fpx.ssm_regression_wiener_velocity(ts, impl=impl_conv, dim=2)
    ssm_conv = ssm_parametrize(noise=1e-4, diffusion=1.0)

    # Sample using the conventional parametrisation
    latent, data = _sample(ssm=ssm_conv, impl=impl_conv)
    assert latent.shape == (len(ts), 4)
    assert data.shape == (len(ts) - 1, 2)

    # Replicate with sqrt parametrisation
    impl_sqrt = fpx.impl_cholesky_based()
    ssm_parametrize = fpx.ssm_regression_wiener_velocity(ts, impl=impl_sqrt, dim=2)
    ssm_sqrt = ssm_parametrize(noise=1e-4, diffusion=1.0)

    compute_conv = fpx.compute_filter(impl=impl_conv)
    compute_sqrt = fpx.compute_filter(impl=impl_sqrt)

    rv_conv, _aux = compute_conv(data, ssm_conv)
    rv_sqrt, _aux = compute_sqrt(data, ssm_sqrt)

    rv_conv = impl_conv.rv_to_mvnorm(rv_conv)
    rv_sqrt = impl_sqrt.rv_to_mvnorm(rv_sqrt)

    for x1, x2 in zip(jax.tree.leaves(rv_conv), jax.tree.leaves(rv_sqrt)):
        assert allclose(x1, x2)


def allclose(a, b):
    atol = jnp.sqrt(jnp.finfo(a.dtype).eps)
    rtol = jnp.sqrt(jnp.finfo(a.dtype).eps)
    return jnp.allclose(a, b, atol=atol, rtol=rtol)


def _sample(*, ssm, impl):
    key = jax.random.PRNGKey(seed=1)
    sample = fpx.compute_stats_sample(impl=impl)
    (latent, data) = sample(key, ssm)
    return latent, data


def rmse(a, b):
    return jnp.linalg.norm(a - b) / jnp.sqrt(a.size)
