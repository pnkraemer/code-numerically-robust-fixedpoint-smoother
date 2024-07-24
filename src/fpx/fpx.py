"""FPX: Fixed-point smoothing in JAX."""

import dataclasses
import jax.numpy as jnp
import jax
from typing import Callable, NamedTuple, Any, TypeVar, Generic


T = TypeVar("T")


class GaussCond(NamedTuple):
    """Affine Gaussian conditional distributions."""

    transition: jax.Array
    noise: Any


@dataclasses.dataclass
class Impl(Generic[T]):
    """State-space model implementation."""

    rv_initialize: Callable[[jax.Array, jax.Array], T]
    rv_sample: Callable[[Any, T], jax.Array]
    parametrize_conditional: Callable[[jax.Array, GaussCond], T]
    marginalize: Callable[[T, GaussCond], T]
    bayes_update: Callable[[T, GaussCond], tuple[T, GaussCond]]


def impl_conventional() -> Impl:
    """Construct a state-space model implementation."""

    def rv_sample(key, rv):
        mean, cov = rv
        base = jax.random.normal(key, shape=mean.shape, dtype=mean.dtype)
        return mean + jnp.linalg.cholesky(cov) @ base

    def parametrize_conditional(x, /, ssm):
        A, (mean, cov) = ssm
        return A @ x + mean, cov

    def marginalize(rv, /, ssm):
        A, (mean, cov) = ssm
        return A @ rv[0] + mean, A @ rv[1] @ A.T + cov

    def bayes_update(rv, ssm):
        H, (r, R) = ssm
        m, C = rv

        s = H @ m + r
        S = H @ C @ H.T + R

        gain = C @ H.T @ jnp.linalg.inv(S)
        mean_new = m - gain @ s
        cov_new = C - gain @ S @ gain.T
        return (s, S), GaussCond(gain, (mean_new, cov_new))

    return Impl(
        rv_initialize=lambda m, c: (m, c),
        rv_sample=rv_sample,
        parametrize_conditional=parametrize_conditional,
        marginalize=marginalize,
        bayes_update=bayes_update,
    )


class Dynamics(NamedTuple):
    """State-space model dynamics."""

    latent: GaussCond
    observation: GaussCond


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
        return Dynamics(latent=GaussCond(A, rv_q), observation=GaussCond(H, rv_r))

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
class Algorithm(Generic[T]):
    init: Callable[[Any], T]
    predict: Callable[[T, GaussCond], T]
    update: Callable[[T, GaussCond, jax.Array], T]
    extract: Callable[[T], Any]


def alg_filter_kalman(impl: Impl) -> Algorithm:
    def update(rv, model, data):
        _rv, cond = impl.bayes_update(rv, model)
        return impl.parametrize_conditional(data, cond)

    return Algorithm(
        init=lambda x: x,
        extract=lambda x: x,
        predict=impl.marginalize,
        update=update,
    )


def estimate_state(data: jax.Array, init, ssm: SSM, algorithm: Algorithm):
    def step_fun(state_k: T, inputs: tuple[jax.Array, Dynamics]) -> tuple[T, Any]:
        # Read
        (y_k, model_k) = inputs

        # Predict
        state_kplus = algorithm.predict(state_k, model_k.latent)

        # Update
        state_new = algorithm.update(state_kplus, model_k.observation, y_k)
        return state_new, algorithm.extract(state_new)

    x0 = algorithm.init(init)
    return jax.lax.scan(step_fun, xs=(data, ssm), init=x0)
