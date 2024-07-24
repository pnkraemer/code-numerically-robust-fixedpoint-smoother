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
    _, (latent, data) = fpx.sample_sequence(key, x0, ssm.dynamics, impl=impl)
    assert latent.shape == (len(ts) - 1, 4)
    assert data.shape == (len(ts) - 1, 2)

    # Run a Kalman filter
    filter_kalman = fpx.alg_filter_kalman(impl=impl)
    (mean, cov), _aux = fpx.estimate_state(data, ssm, algorithm=filter_kalman)

    # Assert that the error's magnitude is of the same order
    # as the observation noise
    assert rmse(mean[:2], latent[-1, :2]) < 1e-3


#
# @pytest_cases.parametrize_with_cases("impl", cases=".")
# def test_smoother_more_accurate_than_filter(impl):
#     # Set up a test problem
#     ts = jnp.linspace(0, 1)
#     init, model = fpx.ssm_car_tracking_velocity(
#         ts, noise=1e-4, diffusion=1.0, impl=impl
#     )
#
#     # Create some data
#     key = jax.random.PRNGKey(seed=1)
#     key, subkey = jax.random.split(key, num=2)
#     x0 = impl.rv_sample(subkey, init)
#     _, (latent, data) = fpx.sample_sequence(key, x0, model, impl=impl)
#     assert latent.shape == (len(ts) - 1, 4)
#     assert data.shape == (len(ts) - 1, 2)
#
#     # Run a Kalman filter
#     filter_kalman = fpx.alg_filter_kalman(impl=impl)
#     _, (mean, _cov) = fpx.estimate_state(data, init, model, algorithm=filter_kalman)
#
#     # Assert that the error's magnitude is of the same order
#     # as the observation noise
#     assert rmse(mean[:, :2], latent[:, :2]) < 1e-3
#

# todo: implement a smoother that improves on the RMSE of the filter
# todo: implement a fixed-point smoother that matches the output of the smoother
# todo: implement state-augmentation for the filter that matches the fixed-point smoother
# todo: implement all these methods in sqrt-form


def rmse(a, b):
    return jnp.linalg.norm(a - b) / jnp.sqrt(a.size)
