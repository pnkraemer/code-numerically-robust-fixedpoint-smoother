"""FPX: Fixed-point smoothing in JAX."""

import dataclasses
import jax.numpy as jnp
import jax
from typing import Callable, NamedTuple, Any, TypeVar, Generic


T = TypeVar("T")


class Cond(NamedTuple):
    """Affine Gaussian conditional distributions."""

    A: jax.Array
    noise: Any


class Dynamics(NamedTuple):
    """State-space model dynamics."""

    # todo: make this type generic (and register as pytree)

    latent: Cond
    observation: Cond


@dataclasses.dataclass
class Impl(Generic[T]):
    """State-space model implementation."""

    rv_initialize: Callable[[jax.Array, jax.Array], T]
    rv_sample: Callable[[Any, T], jax.Array]
    rv_fixedpoint_augment: Callable[[T], T]
    rv_fixedpoint_select: Callable[[T], T]
    dynamics_fixedpoint: Callable[[Dynamics], Dynamics]
    parametrize_conditional: Callable[[jax.Array, Cond], T]
    marginalize: Callable[[T, Cond], T]
    bayes_update: Callable[[T, Cond], tuple[T, Cond]]


def impl_conventional() -> Impl:
    """Construct a state-space model implementation."""

    class Normal(NamedTuple):
        mean: jax.Array
        cov: jax.Array

    def rv_sample(key, /, rv: Normal) -> jax.Array:
        base = jax.random.normal(key, shape=rv.mean.shape, dtype=rv.mean.dtype)
        return rv.mean + jnp.linalg.cholesky(rv.cov) @ base

    def parametrize_conditional(x: jax.Array, /, cond: Cond) -> Normal:
        return Normal(cond.A @ x + cond.noise.mean, cond.noise.cov)

    def marginalize(rv: Normal, /, cond: Cond) -> Normal:
        mean = cond.A @ rv.mean + cond.noise.mean
        cov = cond.A @ rv.cov @ cond.A.T + cond.noise.cov
        return Normal(mean, cov)

    def bayes_update(rv: Normal, cond: Cond) -> tuple[Normal, Cond]:
        s = cond.A @ rv.mean + cond.noise.mean
        S = cond.A @ rv.cov @ cond.A.T + cond.noise.cov

        gain = rv.cov @ cond.A.T @ jnp.linalg.inv(S)
        mean_new = rv.mean - gain @ s
        cov_new = rv.cov - gain @ S @ gain.T
        return Normal(s, S), Cond(gain, Normal(mean_new, cov_new))

    def rv_fixedpoint_augment(rv: Normal) -> Normal:
        mean, cov = rv
        mean_augmented = jnp.concatenate([mean, mean], axis=0)
        cov_augmented_row = jnp.concatenate([cov, cov], axis=1)
        cov_augmented = jnp.concatenate([cov_augmented_row, cov_augmented_row], axis=0)
        return Normal(mean_augmented, cov_augmented)

    def rv_fixedpoint_select(rv: Normal) -> Normal:
        mean, cov = rv
        n = len(mean) // 2
        return Normal(mean[n:], cov[n:, n:])

    def dynamics_fixedpoint(dynamics: Dynamics) -> Dynamics:
        # Augment latent dynamics
        A, (q, Q) = dynamics.latent
        A_ = jax.scipy.linalg.block_diag(A, jnp.eye(len(A)))
        q_ = jnp.concatenate([q, jnp.zeros_like(q)], axis=0)
        Q_ = jax.scipy.linalg.block_diag(Q, jnp.zeros_like(Q))
        latent = Cond(A_, Normal(q_, Q_))

        # Augment observations
        H, cond = dynamics.observation
        H_ = jnp.concatenate([H, jnp.zeros_like(H)], axis=1)
        observation = Cond(H_, cond)

        # Combine and return
        return Dynamics(latent=latent, observation=observation)

    return Impl(
        rv_initialize=lambda m, c: Normal(m, c),
        rv_sample=rv_sample,
        parametrize_conditional=parametrize_conditional,
        marginalize=marginalize,
        bayes_update=bayes_update,
        rv_fixedpoint_augment=rv_fixedpoint_augment,
        rv_fixedpoint_select=rv_fixedpoint_select,
        dynamics_fixedpoint=dynamics_fixedpoint,
    )


class SSM(NamedTuple):
    """State-space model."""

    # todo: make this type generic (and register as pytree)
    init: Any
    dynamics: Dynamics


def ssm_augment_fixedpoint(ssm: SSM, impl: Impl) -> SSM:
    init = impl.rv_fixedpoint_augment(ssm.init)
    dynamics = jax.vmap(impl.dynamics_fixedpoint)(ssm.dynamics)
    return SSM(init=init, dynamics=dynamics)


def ssm_car_tracking_velocity(ts, /, noise, diffusion, impl: Impl) -> SSM:
    """Construct a Wiener-velocity car-tracking model."""

    def transition(dt) -> Dynamics:
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
        return Dynamics(latent=Cond(A, rv_q), observation=Cond(H, rv_r))

    m0 = jnp.zeros((4,))
    C0 = jnp.eye(4)
    x0 = impl.rv_initialize(m0, C0)

    return SSM(init=x0, dynamics=jax.vmap(transition)(jnp.diff(ts)))


def sequence_sample(key, x0: jax.Array, dynamics: Dynamics, impl: Impl):
    def scan_fun(x, dynamics_k: Dynamics):
        key_k, sample_k = x

        key_k, subkey_k = jax.random.split(key_k, num=2)
        rv = impl.parametrize_conditional(sample_k, dynamics_k.latent)
        sample_k = impl.rv_sample(subkey_k, rv)

        key_k, subkey_k = jax.random.split(key_k, num=2)
        rv_obs = impl.parametrize_conditional(sample_k, dynamics_k.observation)
        sample_obs_k = impl.rv_sample(subkey_k, rv_obs)
        return (key_k, sample_k), (sample_k, sample_obs_k)

    return jax.lax.scan(scan_fun, xs=dynamics, init=(key, x0))


def sequence_marginalize(init, cond: Cond, impl: Impl, reverse: bool):
    def scan_fun(x, cond_k: Cond):
        marg = impl.marginalize(x, cond_k)
        return marg, marg

    return jax.lax.scan(scan_fun, xs=cond, init=init, reverse=reverse)


@dataclasses.dataclass
class Algorithm(Generic[T]):
    init: Callable[[Any], T]
    predict: Callable[[T, Cond], T]
    update: Callable[[T, Cond, jax.Array], T]
    extract: Callable[[T], Any]


def alg_fixedpoint_via_filter(algorithm_filter: Algorithm, impl: Impl) -> Algorithm:
    def extract(solution):
        rv_final, aux = algorithm_filter.extract(solution)
        rv_reduced = impl.rv_fixedpoint_select(rv_final)
        return rv_reduced, {"rv_final": rv_final, "filter_aux": aux}

    return Algorithm(
        init=algorithm_filter.init,
        predict=algorithm_filter.predict,
        update=algorithm_filter.update,
        extract=extract,
    )


def alg_filter_kalman(impl: Impl) -> Algorithm:
    def update(rv, model, data):
        _rv, cond = impl.bayes_update(rv, model)
        return impl.parametrize_conditional(data, cond)

    def extract(solution):
        rv_final, rv_intermediates = solution
        return rv_final, {"intermediates": rv_intermediates}

    return Algorithm(
        init=lambda x: x,
        extract=extract,
        predict=impl.marginalize,
        update=update,
    )


def alg_smoother_rts(impl: Impl) -> Algorithm:
    def init(rv):
        eye = jnp.eye(len(rv.mean))
        zeros = jax.tree.map(jnp.zeros_like, rv)
        return rv, Cond(eye, zeros)

    def predict(x, cond):
        return impl.bayes_update(x[0], cond)

    def update(x, model, data):
        rv, cond_posterior = x
        _rv, cond = impl.bayes_update(rv, model)
        return impl.parametrize_conditional(data, cond), cond_posterior

    def extract(solution):
        _terminal, (filter_solution, conds) = solution
        rv = jax.tree.map(lambda s: s[-1, ...], filter_solution)
        return (rv, conds), {"filter_solution": filter_solution}

    return Algorithm(
        init=init,
        extract=extract,
        predict=predict,
        update=update,
    )


# todo: this code assumes $p(x_0 \mid y_{1:K})$,
#  and we should use the same notation in the paper
#  this explanation is superior,
#  because it saves the initialisation step in all algorithm boxes!
def estimate_state(data: jax.Array, ssm: SSM, *, algorithm: Algorithm):
    def step_fun(state_k: T, inputs: tuple[jax.Array, Dynamics]) -> tuple[T, Any]:
        # Read
        (y_k, model_k) = inputs

        # Predict
        state_kplus = algorithm.predict(state_k, model_k.latent)

        # Update
        state_new = algorithm.update(state_kplus, model_k.observation, y_k)
        return state_new, state_new

    x0 = algorithm.init(ssm.init)
    solution = jax.lax.scan(step_fun, xs=(data, ssm.dynamics), init=x0)
    return algorithm.extract(solution)
