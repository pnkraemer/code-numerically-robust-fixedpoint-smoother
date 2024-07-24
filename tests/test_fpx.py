"""Test the Kalman filter implementation."""

import pytest_cases
import jax.numpy as jnp
import jax

from fpx import fpx


def case_ssm_conventional():
    return fpx.ssm_conventional()


@pytest_cases.parametrize_with_cases("ssm", cases=".")
def test_filter_estimates_trajectory_accurately(ssm):

    # Set up a test problem
    ts = jnp.linspace(0, 1)
    init, model = fpx.model_car_tracking_velocity(
        ts, noise=1e-4, diffusion=1.0, ssm=ssm
    )

    # Create some data
    key = jax.random.PRNGKey(seed=1)
    key, subkey = jax.random.split(key, num=2)
    x0 = ssm.sample(subkey, init)
    _, (latent, data) = fpx.sample(key, x0, model, ssm=ssm)
    assert latent.shape == (len(ts) - 1, 4)
    assert data.shape == (len(ts) - 1, 2)

    # Run a Kalman filter
    filter_kalman = fpx.alg_filter_kalman(ssm=ssm)
    _, (mean, _cov) = fpx.estimate_state(data, init, model, algorithm=filter_kalman)

    # Assert that the error's magnitude is of the same order
    # as the observation noise
    assert rmse(mean[:, :2], latent[:, :2]) < 1e-3


# todo: implement a smoother that improves on the RMSE of the filter
# todo: implement a fixed-point smoother that matches the output of the smoother
# todo: implement state-augmentation for the filter that matches the fixed-point smoother
# todo: implement all these methods in sqrt-form


def rmse(a, b):
    return jnp.linalg.norm(a - b) / jnp.sqrt(a.size)
