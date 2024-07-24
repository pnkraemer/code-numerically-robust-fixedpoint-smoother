"""Test the Kalman filter implementation."""

import pytest_cases
import jax.numpy as jnp
import jax

from fpx import fpx


def case_impl_conventional():
    return fpx.impl_conventional()


@pytest_cases.parametrize_with_cases("impl", cases=".")
def test_filter_estimates_trajectory_accurately(impl):
    # Set up a test problem
    ts = jnp.linspace(0, 1)
    ssm = fpx.ssm_car_tracking_velocity(ts, noise=1e-4, diffusion=1.0, impl=impl)

    # Create some data
    key = jax.random.PRNGKey(seed=1)
    key, subkey = jax.random.split(key, num=2)
    x0 = impl.rv_sample(subkey, ssm.init)
    _, (latent, data) = fpx.sequence_sample(key, x0, ssm.dynamics, impl=impl)
    assert latent.shape == (len(ts) - 1, 4)
    assert data.shape == (len(ts) - 1, 2)

    # Run a Kalman filter
    (mean, cov), _aux = fpx.estimate_filter_kalman(data, ssm, impl=impl)

    # Assert that the error's magnitude is of the same order
    # as the observation noise
    assert rmse(mean[:2], latent[-1, :2]) < 1e-3


@pytest_cases.parametrize_with_cases("impl", cases=".")
def test_smoother_more_accurate_than_filter(impl):
    # Set up a test problem
    ts = jnp.linspace(0, 1)
    ssm = fpx.ssm_car_tracking_velocity(ts, noise=1e-1, diffusion=1.0, impl=impl)

    # Create some data
    key = jax.random.PRNGKey(seed=1)
    key, subkey = jax.random.split(key, num=2)
    x0 = impl.rv_sample(subkey, ssm.init)
    _, (latent, data) = fpx.sequence_sample(key, x0, ssm.dynamics, impl=impl)
    assert latent.shape == (len(ts) - 1, 4)
    assert data.shape == (len(ts) - 1, 2)

    # Run a Kalman filter
    terminal_filter, _aux = fpx.estimate_filter_kalman(data, ssm, impl=impl)

    # Run an RTS smoother
    (terminal, conds), aux = fpx.estimate_smoother_rts(data, ssm, impl=impl)
    marginals = fpx.sequence_marginalize(terminal, conds, impl=impl, reverse=True)

    # Assert that the final states of filter and smoother coincide
    assert jax.tree.all(jax.tree.map(jnp.allclose, terminal, terminal_filter))

    # Assert that the marginals make sense
    # Select the intermediate states of data, filter, and smoother
    # so that the RMSEs can be compared. Precisely:
    # The smoother-solution does not include the terminal value
    # The filter-solution does not include the initial value
    # The data vectors match the filter solution
    m_f = aux["filter_distributions"][0][:-1]
    y = latent[:-1]
    m_s = marginals.mean[1:]

    # Assert the filter is better than the noise
    # Use '0.9' to increase the significance
    assert rmse(m_f[:, :2], y[:, :2]) < 0.9 * rmse(data[:-1], y[:, :2])

    # Assert that the smoother is better than the filter
    # Use '0.9' to increase the significance
    assert rmse(m_s, y) < 0.9 * rmse(m_f, y)


@pytest_cases.parametrize_with_cases("impl", cases=".")
def test_state_augmented_filter_matches_smoother_at_initial_state(impl):
    # Set up a test problem
    ts = jnp.linspace(0, 1)
    ssm = fpx.ssm_car_tracking_velocity(ts, noise=1e-4, diffusion=1.0, impl=impl)

    # Create some data
    key = jax.random.PRNGKey(seed=1)
    key, subkey = jax.random.split(key, num=2)
    x0 = impl.rv_sample(subkey, ssm.init)
    _, (latent, data) = fpx.sequence_sample(key, x0, ssm.dynamics, impl=impl)
    assert latent.shape == (len(ts) - 1, 4)
    assert data.shape == (len(ts) - 1, 2)

    # Run a fixedpoint-smoother via state-augmented filtering
    # and via marginalising over an RTS solution
    initial_rts, _aux = fpx.estimate_fixedpoint_via_rts(data, ssm, impl=impl)
    initial_fps, _aux = fpx.estimate_fixedpoint_via_filter(data, ssm, impl=impl)

    # Check that all leaves match
    for x1, x2 in zip(jax.tree.leaves(initial_fps), jax.tree.leaves(initial_rts)):
        assert jnp.allclose(x1, x2, atol=1e-4)


# todo: implement a fixed-point smoother that matches the state-augmented filter
# todo: implement all these methods in sqrt-form


def rmse(a, b):
    return jnp.linalg.norm(a - b) / jnp.sqrt(a.size)
