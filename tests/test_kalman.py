"""Test the Kalman filter implementation."""

from fixedpointsmoother import ssm
import pytest_cases
import jax.numpy as jnp
import jax

import matplotlib.pyplot as plt


def case_param_conventional():
    return ssm.param_conventional()


# def case_param_square_root():
#     return ssm.param_square_root()


@pytest_cases.parametrize_with_cases("param", cases=".")
def test_trajectory_estimated(param):
    ts = jnp.linspace(0, 1)

    init, model = ssm.model_car_tracking_velocity(
        ts, noise=1e-2, diffusion=1.0, param=param
    )

    key = jax.random.PRNGKey(seed=1)

    _, (_, observations) = ssm.sample(key, init, model, param=param)
    plt.plot(observations[:, 0], observations[:, 1], "o-")
    plt.show()

    assert False
