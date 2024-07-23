"""Test the Kalman filter implementation."""

import pytest_cases
import jax.numpy as jnp
import jax

from fixedpointsmoother import statespace

import matplotlib.pyplot as plt


def case_ssm_conventional():
    return statespace.ssm_conventional()


# def case_ssm_square_root():
#     return ssm.ssm_square_root()


@pytest_cases.parametrize_with_cases("ssm", cases=".")
def test_trajectory_estimated(ssm):
    ts = jnp.linspace(0, 1)

    init, model = statespace.model_car_tracking_velocity(
        ts, noise=1e-2, diffusion=1.0, ssm=ssm
    )

    key = jax.random.PRNGKey(seed=1)

    _, (_, observations) = statespace.sample(key, init, model, ssm=ssm)
    plt.plot(observations[:, 0], observations[:, 1], "o-")
    plt.show()

    assert False
