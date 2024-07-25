"""Numerically stable fixed-point smoothing in JAX."""

import dataclasses
import jax.numpy as jnp
import jax
from typing import Callable, Any, TypeVar, Generic, NamedTuple, Union


class CholNormal(NamedTuple):
    """Cholesky-based parameters of a multivariate Normal distribution."""

    mean: jax.Array
    cholesky: jax.Array


class CovNormal(NamedTuple):
    """Covariance-based parameters of a multivariate Normal distribution."""

    mean: jax.Array
    cov: jax.Array


T = TypeVar("T", bound=Union[CholNormal, CovNormal])


@dataclasses.dataclass
class Cond(Generic[T]):
    """Affine Gaussian conditional distributions."""

    A: jax.Array
    noise: T


jax.tree_util.register_pytree_node(
    Cond,
    lambda cond: ((cond.A, cond.noise), ()),
    lambda a, c: Cond(*c),
)


@dataclasses.dataclass
class Dynamics(Generic[T]):
    """State-space model dynamics."""

    latent: Cond[T]
    observation: Cond[T]


jax.tree_util.register_pytree_node(
    Dynamics,
    lambda dyn: ((dyn.latent, dyn.observation), ()),
    lambda a, c: Dynamics(*c),
)


@dataclasses.dataclass(frozen=True)
class Impl(Generic[T]):
    """Implementation of estimation in state-space models."""

    rv_from_mvnorm: Callable[[jax.Array, jax.Array], T]
    rv_to_mvnorm: Callable[[T], tuple[jax.Array, jax.Array]]
    rv_sample: Callable[[Any, T], jax.Array]
    rv_fixedpoint_augment: Callable[[T], T]
    rv_fixedpoint_select: Callable[[T], T]
    conditional_parametrize: Callable[[jax.Array, Cond[T]], T]
    conditional_merge: Callable[[Cond[T], Cond[T]], Cond[T]]
    conditional_from_identity: Callable[[T], Cond[T]]
    dynamics_fixedpoint_augment: Callable[[Dynamics[T]], Dynamics[T]]
    marginalize: Callable[[T, Cond[T]], T]
    bayes_update: Callable[[T, Cond[T]], tuple[T, Cond[T]]]


def impl_cholesky_based() -> Impl[CholNormal]:
    """Construct a Cholesky-based implementation of estimation in state-space models."""

    def rv_from_mvnorm(m, c) -> CholNormal:
        return CholNormal(m, jnp.linalg.cholesky(c))

    def rv_to_mvnorm(rv: CholNormal):
        return rv.mean, rv.cholesky @ rv.cholesky.T

    def marginalize(rv: CholNormal, cond: Cond[CholNormal]) -> CholNormal:
        mean = cond.A @ rv.mean + cond.noise.mean

        R_X_F = rv.cholesky.T @ cond.A.T
        R_X = cond.noise.cholesky.T
        cholesky_marg_T = jnp.concatenate([R_X_F, R_X], axis=0)
        cholesky_T = jnp.linalg.qr(cholesky_marg_T, mode="r")
        return CholNormal(mean, cholesky_T.T)

    def bayes_update(rv: CholNormal, cond: Cond[CholNormal]):
        R_YX = cond.noise.cholesky.T
        R_X = rv.cholesky.T
        R_X_F = rv.cholesky.T @ cond.A.T
        R_y, (R_xy, G) = _revert_conditional(R_X_F=R_X_F, R_X=R_X, R_YX=R_YX)

        s = cond.A @ rv.mean + cond.noise.mean
        mean_new = rv.mean - G @ s
        return CholNormal(s, R_y.T), Cond(G, CholNormal(mean_new, R_xy.T))

    def conditional_parametrize(x: jax.Array, cond: Cond[CholNormal]):
        return CholNormal(cond.A @ x + cond.noise.mean, cond.noise.cholesky)

    def rv_sample(key, /, rv: CholNormal) -> jax.Array:
        base = jax.random.normal(key, shape=rv.mean.shape, dtype=rv.mean.dtype)
        return rv.mean + rv.cholesky @ base

    def conditional_merge(
        cond1: Cond[CholNormal], cond2: Cond[CholNormal]
    ) -> Cond[CholNormal]:
        A1, (m1, C1) = cond1.A, cond1.noise
        A2, (m2, C2) = cond2.A, cond2.noise

        A = A1 @ A2
        m = A1 @ m2 + m1
        chol_T = jnp.linalg.qr(jnp.concatenate([C2.T @ A1.T, C1.T]), mode="r")
        return Cond(A, CholNormal(m, chol_T.T))

    def conditional_from_identity(like: CholNormal) -> Cond[CholNormal]:
        eye = jnp.eye(len(like.mean))
        noise = jax.tree.map(jnp.zeros_like, like)
        return Cond(eye, noise)

    def rv_fixedpoint_augment(rv: CholNormal) -> CholNormal:
        mean, chol = rv
        mean_augmented = jnp.concatenate([mean, mean], axis=0)
        chol_augmented = jnp.block(
            [[chol, jnp.zeros_like(chol)], [chol, jnp.zeros_like(chol)]]
        )
        return CholNormal(mean_augmented, chol_augmented)

    def dynamics_fixedpoint_augment(
        dynamics: Dynamics[CholNormal],
    ) -> Dynamics[CholNormal]:
        # Augment latent dynamics
        A, (q, Q) = dynamics.latent.A, dynamics.latent.noise
        A_ = jax.scipy.linalg.block_diag(A, jnp.eye(len(A)))
        q_ = jnp.concatenate([q, jnp.zeros_like(q)], axis=0)
        Q_ = jax.scipy.linalg.block_diag(Q, jnp.zeros_like(Q))
        latent = Cond(A_, CholNormal(q_, Q_))

        # Augment observations
        H, cond = dynamics.observation.A, dynamics.observation.noise
        H_ = jnp.concatenate([H, jnp.zeros_like(H)], axis=1)
        observation = Cond(H_, cond)

        # Combine and return
        return Dynamics(latent=latent, observation=observation)

    def rv_fixedpoint_select(rv: CholNormal) -> CholNormal:
        mean, cholesky = rv
        n = len(mean) // 2
        mean_new = mean[n:]
        cholesky_new = jnp.linalg.qr(cholesky.T[:, n:], mode="r").T
        return CholNormal(mean_new, cholesky_new)

    return Impl(
        rv_from_mvnorm=rv_from_mvnorm,
        rv_to_mvnorm=rv_to_mvnorm,
        rv_sample=rv_sample,
        rv_fixedpoint_select=rv_fixedpoint_select,
        rv_fixedpoint_augment=rv_fixedpoint_augment,
        dynamics_fixedpoint_augment=dynamics_fixedpoint_augment,
        conditional_parametrize=conditional_parametrize,
        conditional_merge=conditional_merge,
        marginalize=marginalize,
        bayes_update=bayes_update,
        conditional_from_identity=conditional_from_identity,
    )


def _revert_conditional(*, R_X_F: jax.Array, R_X: jax.Array, R_YX: jax.Array):
    # Taken from:
    # https://github.com/pnkraemer/probdiffeq/blob/main/probdiffeq/util/cholesky_util.py

    R = jnp.block([[R_YX, jnp.zeros((R_YX.shape[0], R_X.shape[1]))], [R_X_F, R_X]])
    R = jnp.linalg.qr(R, mode="r")

    # ~R_{Y}
    d_out = R_YX.shape[1]
    R_Y = R[:d_out, :d_out]

    # something like the cross-covariance
    R12 = R[:d_out, d_out:]

    # Implements G = R12.T @ np.linalg.inv(R_Y.T) in clever:
    # G = R12.T @ jnp.linalg.inv(R_Y.T)
    G = jax.scipy.linalg.solve_triangular(R_Y, R12, lower=False).T

    # ~R_{X \mid Y}
    R_XY = R[d_out:, d_out:]
    return R_Y, (R_XY, G)


def impl_covariance_based() -> Impl[CovNormal]:
    """Construct a covariance-based implementation of estimation in state-space models."""

    def rv_sample(key, /, rv: CovNormal) -> jax.Array:
        base = jax.random.normal(key, shape=rv.mean.shape, dtype=rv.mean.dtype)
        return rv.mean + jnp.linalg.cholesky(rv.cov) @ base

    def conditional_parametrize(x: jax.Array, /, cond: Cond[CovNormal]) -> CovNormal:
        return CovNormal(cond.A @ x + cond.noise.mean, cond.noise.cov)

    def marginalize(rv: CovNormal, /, cond: Cond[CovNormal]) -> CovNormal:
        mean = cond.A @ rv.mean + cond.noise.mean
        cov = cond.A @ rv.cov @ cond.A.T + cond.noise.cov
        return CovNormal(mean, cov)

    def bayes_update(
        rv: CovNormal, cond: Cond[CovNormal]
    ) -> tuple[CovNormal, Cond[CovNormal]]:
        s = cond.A @ rv.mean + cond.noise.mean
        S = cond.A @ rv.cov @ cond.A.T + cond.noise.cov

        gain = jnp.linalg.solve(S.T, cond.A @ rv.cov).T
        mean_new = rv.mean - gain @ s
        cov_new = rv.cov - gain @ S @ gain.T
        return CovNormal(s, S), Cond(gain, CovNormal(mean_new, cov_new))

    def rv_fixedpoint_augment(rv: CovNormal) -> CovNormal:
        mean, cov = rv
        mean_augmented = jnp.concatenate([mean, mean], axis=0)
        cov_augmented_row = jnp.concatenate([cov, cov], axis=1)
        cov_augmented = jnp.concatenate([cov_augmented_row, cov_augmented_row], axis=0)
        return CovNormal(mean_augmented, cov_augmented)

    def rv_fixedpoint_select(rv: CovNormal) -> CovNormal:
        mean, cov = rv
        n = len(mean) // 2
        return CovNormal(mean[n:], cov[n:, n:])

    def dynamics_fixedpoint_augment(
        dynamics: Dynamics[CovNormal],
    ) -> Dynamics[CovNormal]:
        # Augment latent dynamics
        A, (q, Q) = dynamics.latent.A, dynamics.latent.noise
        A_ = jax.scipy.linalg.block_diag(A, jnp.eye(len(A)))
        q_ = jnp.concatenate([q, jnp.zeros_like(q)], axis=0)
        Q_ = jax.scipy.linalg.block_diag(Q, jnp.zeros_like(Q))
        latent = Cond(A_, CovNormal(q_, Q_))

        # Augment observations
        H, cond = dynamics.observation.A, dynamics.observation.noise
        H_ = jnp.concatenate([H, jnp.zeros_like(H)], axis=1)
        observation = Cond(H_, cond)

        # Combine and return
        return Dynamics(latent=latent, observation=observation)

    def conditional_merge(
        cond1: Cond[CovNormal], cond2: Cond[CovNormal]
    ) -> Cond[CovNormal]:
        A1, (m1, C1) = cond1.A, cond1.noise
        A2, (m2, C2) = cond2.A, cond2.noise

        A = A1 @ A2
        m = A1 @ m2 + m1
        C = A1 @ C2 @ A1.T + C1
        return Cond(A, CovNormal(m, C))

    def conditional_from_identity(like: CovNormal) -> Cond[CovNormal]:
        eye = jnp.eye(len(like.mean))
        noise = jax.tree.map(jnp.zeros_like, like)
        return Cond(eye, noise)

    return Impl(
        rv_to_mvnorm=lambda rv: (rv.mean, rv.cov),
        rv_from_mvnorm=lambda m, c: CovNormal(m, c),
        rv_sample=rv_sample,
        conditional_parametrize=conditional_parametrize,
        marginalize=marginalize,
        bayes_update=bayes_update,
        rv_fixedpoint_augment=rv_fixedpoint_augment,
        rv_fixedpoint_select=rv_fixedpoint_select,
        dynamics_fixedpoint_augment=dynamics_fixedpoint_augment,
        conditional_merge=conditional_merge,
        conditional_from_identity=conditional_from_identity,
    )


@dataclasses.dataclass
class SSM(Generic[T]):
    """State-space model parametrisation."""

    init: T
    dynamics: Dynamics[T]


jax.tree_util.register_pytree_node(
    SSM,
    lambda ssm: ((ssm.init, ssm.dynamics), ()),
    lambda a, c: SSM(*c),
)


def ssm_car_tracking_velocity(
    ts, /, noise, diffusion, impl: Impl[T], dim: int = 2
) -> SSM[T]:
    """Construct a Wiener-velocity car-tracking model."""

    def transition(dt) -> Dynamics:
        eye_d = jnp.eye(dim)
        one_d = jnp.ones((dim,))

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

        rv_q = impl.rv_from_mvnorm(q, Q)
        rv_r = impl.rv_from_mvnorm(r, R)
        return Dynamics(latent=Cond(A, rv_q), observation=Cond(H, rv_r))

    m0 = jnp.zeros((2 * dim,))
    C0 = jnp.eye(2 * dim)
    x0 = impl.rv_from_mvnorm(m0, C0)

    return SSM(init=x0, dynamics=jax.vmap(transition)(jnp.diff(ts)))


def ssm_car_tracking_acceleration(
    ts, /, noise, diffusion, impl: Impl[T], dim: int = 2
) -> SSM[T]:
    """Construct a Wiener-acceleration car-tracking model."""

    def transition(dt) -> Dynamics:
        eye_d = jnp.eye(dim)
        one_d = jnp.ones((dim,))

        A_1d = jnp.asarray([[1.0, dt, dt**2 / 2], [0, 1.0, dt], [0, 0, 1.0]])
        q_1d = jnp.asarray([0.0, 0.0, 0.0])
        Q_1d = diffusion**2 * jnp.asarray(
            [
                [dt**5 / 20, dt**4 / 8, dt**3 / 6],
                [dt**4 / 8, dt**3 / 3, dt**2 / 2],
                [dt**3 / 3, dt**2 / 2, dt],
            ]
        )
        Q_1d += 1e-8 * jnp.eye(len(Q_1d))

        H_1d = jnp.asarray([[1.0, 0.0, 0.0]])
        r_1d = jnp.asarray(0.0)
        R_1d = noise**2 * jnp.asarray(1.0)

        A = jnp.kron(A_1d, eye_d)
        q = jnp.kron(q_1d, one_d)
        Q = jnp.kron(Q_1d, eye_d)

        H = jnp.kron(H_1d, eye_d)
        r = jnp.kron(r_1d, one_d)
        R = jnp.kron(R_1d, eye_d)

        rv_q = impl.rv_from_mvnorm(q, Q)
        rv_r = impl.rv_from_mvnorm(r, R)
        return Dynamics(latent=Cond(A, rv_q), observation=Cond(H, rv_r))

    m0 = jnp.zeros((3 * dim,))
    C0 = jnp.eye(3 * dim)
    x0 = impl.rv_from_mvnorm(m0, C0)
    dynamics = jax.vmap(transition)(jnp.diff(ts))
    return SSM(init=x0, dynamics=dynamics)


def sequence_sample(key, x0: jax.Array, dynamics: Dynamics[T], impl: Impl[T]):
    """Sample from a state-space model (sequentially)."""

    def scan_fun(x, dynamics_k: Dynamics):
        key_k, sample_k = x

        key_k, subkey_k = jax.random.split(key_k, num=2)
        rv = impl.conditional_parametrize(sample_k, dynamics_k.latent)
        sample_k = impl.rv_sample(subkey_k, rv)

        key_k, subkey_k = jax.random.split(key_k, num=2)
        rv_obs = impl.conditional_parametrize(sample_k, dynamics_k.observation)
        sample_obs_k = impl.rv_sample(subkey_k, rv_obs)
        return (key_k, sample_k), (sample_k, sample_obs_k)

    return jax.lax.scan(scan_fun, xs=dynamics, init=(key, x0))


def sequence_marginalize(init, cond: Cond[T], impl: Impl[T], reverse: bool) -> T:
    """Marginalize a sequence of conditionals (sequentially)."""

    def scan_fun(x, cond_k: Cond[T]):
        marg = impl.marginalize(x, cond_k)
        return marg, marg

    _, all_ = jax.lax.scan(scan_fun, xs=cond, init=init, reverse=reverse)
    return all_


# todo: this code assumes $p(x_0 \mid y_{1:K})$,
#  and we should use the same notation in the paper
#  this explanation is superior,
#  because it saves the initialisation step in all algorithm boxes!


def estimate_fixedpoint(impl: Impl[T]) -> Callable:
    """Estimate a solution of the fixed-point smoothing problem."""

    def estimate(data: jax.Array, ssm: SSM[T]):
        def step_fun(state_k, inputs: tuple[jax.Array, Dynamics]) -> tuple:
            # Read
            (y_k, model_k) = inputs

            # Predict
            state_rv, state_backward = state_k
            state_kplus, backward = impl.bayes_update(state_rv, model_k.latent)
            backward = impl.conditional_merge(state_backward, backward)

            # Update
            _rv, gain = impl.bayes_update(state_kplus, model_k.observation)
            state_new = impl.conditional_parametrize(y_k, gain)
            return (state_new, backward), ()

        cond = impl.conditional_from_identity(ssm.init)
        init, xs = (ssm.init, cond), (data, ssm.dynamics)
        (rv, cond), _ = jax.lax.scan(step_fun, xs=xs, init=init)
        return impl.marginalize(rv, cond), {}

    return estimate


def estimate_fixedpoint_via_fixedinterval(impl: Impl[T]) -> Callable:
    """Estimate a solution of the fixed-point smoothing problem.

    Calls a fixed-interval smoother internally.
    """
    estimate_interval = estimate_fixedinterval(impl=impl)

    def estimate(data: jax.Array, ssm: SSM[T]):
        (terminal, conds), aux = estimate_interval(data=data, ssm=ssm)
        marginals = sequence_marginalize(terminal, conds, impl=impl, reverse=True)
        initial_rts = jax.tree.map(lambda s: s[0, ...], marginals)
        return initial_rts, {}

    return estimate


def estimate_fixedinterval(impl: Impl[T]) -> Callable:
    """Estimate a solution of the fixed-interval smoothing problem."""

    def estimate(data: jax.Array, ssm: SSM[T]):
        def step_fun(state_k: T, inputs: tuple[jax.Array, Dynamics]) -> tuple[T, Any]:
            # Read
            (y_k, model_k) = inputs

            # Predict
            state_kplus, cond = impl.bayes_update(state_k, model_k.latent)

            # Update
            _rv, gain = impl.bayes_update(state_kplus, model_k.observation)
            state_new = impl.conditional_parametrize(y_k, gain)
            return state_new, (state_new, cond)

        x0 = ssm.init
        solution, (filterdists, conds) = jax.lax.scan(
            step_fun, xs=(data, ssm.dynamics), init=x0
        )
        return (solution, conds), {"filter_distributions": filterdists}

    return estimate


def estimate_fixedpoint_via_filter(impl: Impl[T]) -> Callable:
    """Estimate a solution of the fixed-point smoothing problem.

    Augments the state-space model and calls a filter internally.
    """
    estimate_fi = estimate_filter(impl=impl)

    def estimate(data: jax.Array, ssm: SSM[T]):
        ssm_augment = _ssm_augment_fixedpoint(ssm, impl=impl)
        terminal_filter, aux = estimate_fi(data, ssm_augment)
        rv_reduced = impl.rv_fixedpoint_select(terminal_filter)
        return rv_reduced, {}

    return estimate


def _ssm_augment_fixedpoint(ssm: SSM[T], impl: Impl[T]) -> SSM[T]:
    init = impl.rv_fixedpoint_augment(ssm.init)
    dynamics = jax.vmap(impl.dynamics_fixedpoint_augment)(ssm.dynamics)
    return SSM(init=init, dynamics=dynamics)


def estimate_filter(impl: Impl[T]) -> Callable:
    """Estimate a solution of the filtering problem."""

    def estimate(data: jax.Array, ssm: SSM[T]):
        def step_fun(state_k: T, inputs: tuple[jax.Array, Dynamics]) -> tuple[T, Any]:
            # Read
            (y_k, model_k) = inputs

            # Predict
            state_kplus = impl.marginalize(state_k, model_k.latent)

            # Update
            _rv, cond = impl.bayes_update(state_kplus, model_k.observation)
            state_new = impl.conditional_parametrize(y_k, cond)
            return state_new, ()

        x0 = ssm.init
        solution, _ = jax.lax.scan(step_fun, xs=(data, ssm.dynamics), init=x0)
        return solution, {}

    return estimate
