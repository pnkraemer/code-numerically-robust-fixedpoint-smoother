"""State-space models and their parametrisations."""

import dataclasses
import jax.numpy as jnp
import jax
from typing import Callable


@dataclasses.dataclass
class SSM:
    init_rv: Callable
    sample: Callable
    conditional: Callable


def ssm_conventional():
    def sample_(key, rv):
        mean, cov = rv
        base = jax.random.normal(key, shape=mean.shape, dtype=mean.dtype)
        return mean + jnp.linalg.cholesky(cov) @ base

    def conditional(x, /, model):
        A, (mean, cov) = model
        return A @ x + mean, cov

    return SSM(init_rv=lambda *a: a, sample=sample_, conditional=conditional)


def ssm_square_root():
    def init_rv(m, C):
        return m, jnp.linalg.cholesky(C)

    def sample_(key, rv):
        mean, cholesky = rv
        base = jax.random.normal(key, shape=mean.shape, dtype=mean.dtype)
        return mean + cholesky @ base

    def conditional_(x, /, model):
        A, (mean, cholesky) = model
        return (A @ x + mean, cholesky)

    return SSM(init_rv=init_rv, sample=sample_, conditional=conditional_)


def model_car_tracking_velocity(ts, /, noise, diffusion, *, ssm):
    def transition(dt):
        eye_d = jnp.eye(2)
        one_d = jnp.ones((2,))

        A_1d = jnp.asarray([[1.0, dt], [0, 1.0]])
        q_1d = jnp.asarray([0.0, 0.0])
        Q_1d = diffusion**2 * jnp.asarray([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
        H_1d = jnp.asarray([1.0, 0.0])
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


def sample(key, init, model, *, ssm):
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

    key, subkey = jax.random.split(key, num=2)
    x0 = ssm.sample(subkey, init)
    return jax.lax.scan(scan_fun, xs=model, init=(key, x0))
