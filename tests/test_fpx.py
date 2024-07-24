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
    filter_kalman = fpx.alg_filter_kalman(impl=impl)
    (mean, cov), _aux = fpx.estimate_state(data, ssm, algorithm=filter_kalman)

    # Assert that the error's magnitude is of the same order
    # as the observation noise
    assert rmse(mean[:2], latent[-1, :2]) < 1e-3


@pytest_cases.parametrize_with_cases("impl", cases=".")
def test_smoother_more_accurate_than_filter(impl):
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
    filter_kalman = fpx.alg_filter_kalman(impl=impl)
    terminal_filter, aux = fpx.estimate_state(data, ssm, algorithm=filter_kalman)
    mean_filter, _cov = aux["intermediates"]

    # Run an RTS smoother
    smoother_rts = fpx.alg_smoother_rts(impl=impl)
    solution_smoother = fpx.estimate_state(data, ssm, algorithm=smoother_rts)
    (terminal_smoother, conds), _aux = solution_smoother
    marginals_smoother = fpx.sequence_marginalize(
        terminal_smoother, conds, impl=impl, reverse=True
    )
    _initial, (mean_smoother, _cov) = marginals_smoother

    # Assert that the final states coincide
    assert jax.tree.all(jax.tree.map(jnp.allclose, terminal_smoother, terminal_filter))

    # Select the intermediate states of data, filter, and smoother
    # so that the RMSEs can be compared
    m_f = mean_filter[:-1]  # The smoother-solution does not include the terminal value
    y = latent[:-1]  # The smoother-solution does not include the initial value
    m_s = mean_smoother[1:]  # The filter-solution does not include the initial value

    # Assert that the smoother is better than the filter
    assert rmse(m_s, y) < 0.9 * rmse(m_f, y)  # use '0.9' to increase the significance
    assert False


# todo: implement a fixed-point smoother that matches the output of the smoother
# todo: implement state-augmentation for the filter that matches the fixed-point smoother
# todo: implement all these methods in sqrt-form


def rmse(a, b):
    return jnp.linalg.norm(a - b) / jnp.sqrt(a.size)
