"""Test the Kalman filter implementation."""

import pytest_cases
import jax.numpy as jnp
import jax

from fpx import fpx


def case_ssm_conventional():
    return fpx.ssm_conventional()


@pytest_cases.parametrize_with_cases("ssm", cases=".")
def test_trajectory_estimated(ssm):
    ts = jnp.linspace(0, 1)

    init, model = fpx.model_car_tracking_velocity(
        ts, noise=1e-2, diffusion=1.0, ssm=ssm
    )

    key = jax.random.PRNGKey(seed=1)

    key, subkey = jax.random.split(key, num=2)
    x0 = ssm.sample(subkey, init)

    _, (latent, observations) = fpx.sample(key, x0, model, ssm=ssm)
    assert latent.shape == (len(ts) - 1, 4)
    assert observations.shape == (len(ts) - 1, 2)

    _, (mean, _cov) = fpx.filter_kalman(observations, init, model, ssm=ssm)
    print(jax.tree_util.tree_map(jnp.shape, (mean, _cov)))
    print(mean - latent)

    assert False
