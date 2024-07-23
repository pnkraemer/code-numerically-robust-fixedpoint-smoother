"""State-space models and their parametrisations."""

import dataclasses
import jax.numpy as jnp
import jax
from typing import Callable


@dataclasses.dataclass
class SSM:
    init_rv: Callable
    sample: Callable
    conditional: Callable  # todo: rename to parametrize_conditional
    marginal: Callable
    condition: Callable


def ssm_conventional():
    def sample_(key, rv):
        mean, cov = rv
        base = jax.random.normal(key, shape=mean.shape, dtype=mean.dtype)
        return mean + jnp.linalg.cholesky(cov) @ base

    def conditional(x, /, model):
        A, (mean, cov) = model
        return A @ x + mean, cov

    def marginal(rv, /, model):
        A, (mean, cov) = model
        return (A @ rv[0] + mean, A @ rv[1] @ A.T + cov)

    def condition(rv, model, data):
        H, (r, R) = model
        mean, cov = rv
        gain = cov @ H.T @ jnp.linalg.inv(H @ cov @ H.T + R)
        mean_new = mean - gain @ (H @ mean + r - data)
        cov_new = cov - gain @ (H @ cov @ H.T + R) @ gain.T
        return mean_new, cov_new

    return SSM(
        init_rv=lambda *a: a,
        sample=sample_,
        conditional=conditional,
        marginal=marginal,
        condition=condition,
    )


def ssm_square_root():
    def init_rv(m, C):
        return m, jnp.linalg.cholesky(C)

    def sample_(key, rv):
        mean, cholesky = rv
        base = jax.random.normal(key, shape=mean.shape, dtype=mean.dtype)
        return mean + cholesky @ base

    def conditional_(x, /, model):
        A, (mean, cholesky) = model
        return A @ x + mean, cholesky

    return SSM(init_rv=init_rv, sample=sample_, conditional=conditional_)


def model_car_tracking_velocity(ts, /, noise, diffusion, *, ssm: SSM):
    def transition(dt):
        eye_d = jnp.eye(2)
        one_d = jnp.ones((2,))

        A_1d = jnp.asarray([[1.0, dt], [0, 1.0]])
        q_1d = jnp.asarray([0.0, 0.0])
        Q_1d = diffusion**2 * jnp.asarray([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
        H_1d = jnp.asarray([[1.0, 0.0]])
        r_1d = jnp.asarray(0.0)
        R_1d = noise**2 * jnp.asarray(1.0)

        A = jnp.kron(A_1d, eye_d)
        q = jnp.kron(q_1d, one_d)
        Q = jnp.kron(Q_1d, eye_d)

        H = jnp.kron(H_1d, eye_d)
        r = jnp.kron(r_1d, one_d)
        R = jnp.kron(R_1d, eye_d)

        rv_q = ssm.init_rv(q, Q)
        rv_r = ssm.init_rv(r, R)
        return (A, rv_q), (H, rv_r)

    m0 = jnp.zeros((4,))
    C0 = jnp.eye(4)
    x0 = ssm.init_rv(m0, C0)

    return x0, jax.vmap(transition)(jnp.diff(ts))


def sample(key: jax.random.PRNGKey, x0: jax.Array, model, *, ssm: SSM):
    def scan_fun(x, model_k):
        key_k, sample_k = x
        model_prior, model_obs = model_k

        key_k, subkey_k = jax.random.split(key_k, num=2)
        rv = ssm.conditional(sample_k, model_prior)
        sample_k = ssm.sample(subkey_k, rv)

        key_k, subkey_k = jax.random.split(key_k, num=2)
        rv_obs = ssm.conditional(sample_k, model_obs)
        sample_obs_k = ssm.sample(subkey_k, rv_obs)
        return (key_k, sample_k), (sample_k, sample_obs_k)

    return jax.lax.scan(scan_fun, xs=model, init=(key, x0))


@dataclasses.dataclass
class Algorithm:
    init: Callable
    predict: Callable
    update: Callable
    extract: Callable


def alg_filter_kalman(ssm: SSM) -> Algorithm:
    return Algorithm(
        init=lambda x: x,
        extract=lambda x: x,
        predict=ssm.marginal,
        update=ssm.condition,
    )


def estimate_state(data: jax.Array, init, model, algorithm: Algorithm):
    def step_fun(state, inputs):
        # Read
        (y_k, (model_prior, model_obs)) = inputs
        x_k = state

        # Predict
        x_kplus = algorithm.predict(x_k, model_prior)

        # Update
        x_new = algorithm.update(x_kplus, model_obs, y_k)
        return x_new, algorithm.extract(x_new)

    x0 = algorithm.init(init)
    return jax.lax.scan(step_fun, xs=(data, model), init=x0)
