"""FPX: Fixed-point smoothing in JAX."""

import dataclasses
import jax.numpy as jnp
import jax
from typing import Callable, NamedTuple, Any


@dataclasses.dataclass
class Impl:
    """State-space model implementation."""

    rv_initialize: Callable
    rv_sample: Callable
    parametrize_conditional: Callable
    marginalize: Callable
    bayes_update: Callable


def impl_conventional() -> Impl:
    """Construct a state-space model implementation."""

    def compute_sample(key, rv):
        mean, cov = rv
        base = jax.random.normal(key, shape=mean.shape, dtype=mean.dtype)
        return mean + jnp.linalg.cholesky(cov) @ base

    def parametrize_conditional(x, /, ssm):
        A, (mean, cov) = ssm
        return A @ x + mean, cov

    def marginalize(rv, /, ssm):
        A, (mean, cov) = ssm
        return A @ rv[0] + mean, A @ rv[1] @ A.T + cov

    # todo: return the marginal distributions, too, (so we can use the same function for smoothing)
    def bayes_update(rv, ssm, data):
        H, (r, R) = ssm
        mean, cov = rv
        gain = cov @ H.T @ jnp.linalg.inv(H @ cov @ H.T + R)
        mean_new = mean - gain @ (H @ mean + r - data)
        cov_new = cov - gain @ (H @ cov @ H.T + R) @ gain.T
        return mean_new, cov_new

    return Impl(
        rv_initialize=lambda *a: a,
        rv_sample=compute_sample,
        parametrize_conditional=parametrize_conditional,
        marginalize=marginalize,
        bayes_update=bayes_update,
    )


def impl_square_root() -> Impl:
    """Construct a state-space ssm implementation using square-root form."""

    def rv_initialize(m, C):
        return m, jnp.linalg.cholesky(C)

    def rv_sample(key, rv):
        mean, cholesky = rv
        base = jax.random.normal(key, shape=mean.shape, dtype=mean.dtype)
        return mean + cholesky @ base

    def parametrize_conditional(x, /, ssm):
        A, (mean, cholesky) = ssm
        return A @ x + mean, cholesky

    def not_yet(*_a):
        raise NotImplementedError

    return Impl(
        rv_initialize=rv_initialize,
        rv_sample=rv_sample,
        parametrize_conditional=parametrize_conditional,
        bayes_update=not_yet,
        marginalize=not_yet,
    )


class GaussCondAffine(NamedTuple):
    """Affine Gaussian conditional distributions."""

    transition: jax.Array
    noise: Any


class Dynamics(NamedTuple):
    """State-space model dynamics."""

    latent: GaussCondAffine
    observation: GaussCondAffine


class SSM(NamedTuple):
    """State-space model."""

    init: Any
    dynamics: Dynamics


def ssm_car_tracking_velocity(ts, /, noise, diffusion, *, impl: Impl) -> SSM:
    """Construct a Wiener-velocity car-tracking model."""

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

        rv_q = impl.rv_initialize(q, Q)
        rv_r = impl.rv_initialize(r, R)
        return Dynamics(
            latent=GaussCondAffine(A, rv_q), observation=GaussCondAffine(H, rv_r)
        )

    m0 = jnp.zeros((4,))
    C0 = jnp.eye(4)
    x0 = impl.rv_initialize(m0, C0)

    return SSM(x0, jax.vmap(transition)(jnp.diff(ts)))


def sample_sequence(key, x0: jax.Array, ssm: SSM, *, impl: Impl):
    def scan_fun(x, ssm_k):
        key_k, sample_k = x
        ssm_prior, ssm_obs = ssm_k

        key_k, subkey_k = jax.random.split(key_k, num=2)
        rv = impl.parametrize_conditional(sample_k, ssm_prior)
        sample_k = impl.rv_sample(subkey_k, rv)

        key_k, subkey_k = jax.random.split(key_k, num=2)
        rv_obs = impl.parametrize_conditional(sample_k, ssm_obs)
        sample_obs_k = impl.rv_sample(subkey_k, rv_obs)
        return (key_k, sample_k), (sample_k, sample_obs_k)

    return jax.lax.scan(scan_fun, xs=ssm, init=(key, x0))


@dataclasses.dataclass
class Algorithm:
    init: Callable
    predict: Callable
    update: Callable
    extract: Callable


def alg_filter_kalman(impl: Impl) -> Algorithm:
    return Algorithm(
        init=lambda x: x,
        extract=lambda x: x,
        predict=impl.marginalize,
        update=impl.bayes_update,
    )


def estimate_state(data: jax.Array, init, ssm: SSM, algorithm: Algorithm):
    def step_fun(state, inputs):
        # Read
        (y_k, (ssm_prior, ssm_obs)) = inputs
        x_k = state

        # Predict
        x_kplus = algorithm.predict(x_k, ssm_prior)

        # Update
        x_new = algorithm.update(x_kplus, ssm_obs, y_k)
        return x_new, algorithm.extract(x_new)

    x0 = algorithm.init(init)
    return jax.lax.scan(step_fun, xs=(data, ssm), init=x0)
