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

    # todo: group conditional_* methods together (also by name)
    rv_initialize: Callable[[jax.Array, jax.Array], T]
    rv_to_mvnorm: Callable[[T], tuple[jax.Array, jax.Array]]
    rv_sample: Callable[[Any, T], jax.Array]
    rv_fixedpoint_augment: Callable[[T], T]
    rv_fixedpoint_select: Callable[[T], T]
    dynamics_fixedpoint: Callable[[Dynamics], Dynamics]
    parametrize_conditional: Callable[[jax.Array, Cond], T]
    conditional_merge: Callable[[Cond, Cond], Cond]
    marginalize: Callable[[T, Cond], T]
    bayes_update: Callable[[T, Cond], tuple[T, Cond]]


def impl_square_root() -> Impl:
    def not_yet(*_a):
        raise NotImplementedError

    return Impl(
        rv_initialize=not_yet,
        rv_to_mvnorm=not_yet,
        rv_sample=not_yet,
        rv_fixedpoint_select=not_yet,
        rv_fixedpoint_augment=not_yet,
        dynamics_fixedpoint=not_yet,
        parametrize_conditional=not_yet,
        conditional_merge=not_yet,
        marginalize=not_yet,
        bayes_update=not_yet,
    )


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

    def conditional_merge(cond1: Cond, cond2: Cond) -> Cond:
        A1, (m1, C1) = cond1
        A2, (m2, C2) = cond2

        A = A1 @ A2
        m = A1 @ m2 + m1
        C = A1 @ C2 @ A1.T + C1
        return Cond(A, Normal(m, C))

    return Impl(
        rv_to_mvnorm=lambda rv: (rv.mean, rv.cov),
        rv_initialize=lambda m, c: Normal(m, c),
        rv_sample=rv_sample,
        parametrize_conditional=parametrize_conditional,
        marginalize=marginalize,
        bayes_update=bayes_update,
        rv_fixedpoint_augment=rv_fixedpoint_augment,
        rv_fixedpoint_select=rv_fixedpoint_select,
        dynamics_fixedpoint=dynamics_fixedpoint,
        conditional_merge=conditional_merge,
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

    _, all_ = jax.lax.scan(scan_fun, xs=cond, init=init, reverse=reverse)
    return all_


# todo: this code assumes $p(x_0 \mid y_{1:K})$,
#  and we should use the same notation in the paper
#  this explanation is superior,
#  because it saves the initialisation step in all algorithm boxes!


def estimate_filter_kalman(data: jax.Array, ssm: SSM, *, impl: Impl):
    def step_fun(state_k: T, inputs: tuple[jax.Array, Dynamics]) -> tuple[T, Any]:
        # Read
        (y_k, model_k) = inputs

        # Predict
        state_kplus = impl.marginalize(state_k, model_k.latent)

        # Update
        _rv, cond = impl.bayes_update(state_kplus, model_k.observation)
        state_new = impl.parametrize_conditional(y_k, cond)
        return state_new, ()

    x0 = ssm.init
    solution, _ = jax.lax.scan(step_fun, xs=(data, ssm.dynamics), init=x0)
    return solution, {}


def estimate_smoother_rts(data: jax.Array, ssm: SSM, *, impl: Impl):
    def step_fun(state_k: T, inputs: tuple[jax.Array, Dynamics]) -> tuple[T, Any]:
        # Read
        (y_k, model_k) = inputs

        # Predict
        state_kplus, cond = impl.bayes_update(state_k, model_k.latent)

        # Update
        _rv, gain = impl.bayes_update(state_kplus, model_k.observation)
        state_new = impl.parametrize_conditional(y_k, gain)
        return state_new, (state_new, cond)

    x0 = ssm.init
    solution, (filterdists, conds) = jax.lax.scan(
        step_fun, xs=(data, ssm.dynamics), init=x0
    )
    return (solution, conds), {"filter_distributions": filterdists}


def estimate_fixedpoint(data: jax.Array, ssm: SSM, *, impl: Impl):
    def step_fun(state_k, inputs: tuple[jax.Array, Dynamics]) -> tuple:
        # Read
        (y_k, model_k) = inputs

        # Predict
        state_rv, state_backward = state_k
        state_kplus, backward = impl.bayes_update(state_rv, model_k.latent)
        backward = impl.conditional_merge(state_backward, backward)

        # Update
        _rv, gain = impl.bayes_update(state_kplus, model_k.observation)
        state_new = impl.parametrize_conditional(y_k, gain)
        return (state_new, backward), ()

    x0 = ssm.init
    cond = Cond(jnp.eye(len(x0.mean)), jax.tree.map(jnp.zeros_like, x0))
    init, xs = (x0, cond), (data, ssm.dynamics)
    (rv, cond), _ = jax.lax.scan(step_fun, xs=xs, init=init)
    return impl.marginalize(rv, cond), {}


def estimate_fixedpoint_via_filter(data: jax.Array, ssm: SSM, *, impl: Impl):
    ssm_augment = ssm_augment_fixedpoint(ssm, impl=impl)
    terminal_filter, aux = estimate_filter_kalman(data, ssm_augment, impl=impl)
    rv_reduced = impl.rv_fixedpoint_select(terminal_filter)
    return rv_reduced, {}


def estimate_fixedpoint_via_rts(data: jax.Array, ssm: SSM, *, impl: Impl):
    (terminal, conds), aux = estimate_smoother_rts(data, ssm, impl=impl)
    marginals = sequence_marginalize(terminal, conds, impl=impl, reverse=True)
    initial_rts = jax.tree.map(lambda s: s[0, ...], marginals)
    return initial_rts, {}
