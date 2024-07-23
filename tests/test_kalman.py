"""Test the Kalman filter implementation."""

from fixedpointsmoother import ssm
import pytest_cases
import jax.numpy as jnp
import jax


def case_param_conventional():
    return ssm.param_conventional()


def case_param_square_root():
    return ssm.param_square_root()


@pytest_cases.parametrize_with_cases("param", cases=".")
def test_trajectory_estimated(param):
    ts = jnp.linspace(0, 1, 7)

    model = ssm.model_car_tracking_velocity(ts, param=param)
    key = jax.random.PRNGKey(seed=1)
    data = ssm.sample(key, model, param=param)

    print(model)
    print(data)

    assert False
