"""Numerically robust fixed-point smoothing in JAX."""

import dataclasses
from typing import Any, Callable, Generic, NamedTuple, TypeVar

import jax
import jax.flatten_util
import jax.numpy as jnp
import probdiffeq.ivpsolvers
from probdiffeq.impl import impl as impl_probdiffeq

impl_probdiffeq.select("scalar")


class NormalChol(NamedTuple):
    """Cholesky-based parameters of a multivariate Normal distribution."""

    mean: jax.Array
    cholesky: jax.Array


class NormalCov(NamedTuple):
    """Covariance-based parameters of a multivariate Normal distribution."""

    mean: jax.Array
    cov: jax.Array


T = TypeVar("T", bound=NormalChol | NormalCov)
"""A type-variable to indicate random-variable types."""


@dataclasses.dataclass
class SSMCond(Generic[T]):
    """Affine Gaussian conditional distributions."""

    A: jax.Array
    noise: T


jax.tree_util.register_pytree_node(
    SSMCond,
    lambda cond: ((cond.A, cond.noise), ()),
    lambda a, c: SSMCond(*c),
)


@dataclasses.dataclass
class SSMDynamics(Generic[T]):
    """State-space model dynamics."""

    latent: SSMCond[T]
    observation: SSMCond[T]


jax.tree_util.register_pytree_node(
    SSMDynamics,
    lambda dyn: ((dyn.latent, dyn.observation), ()),
    lambda a, c: SSMDynamics(*c),
)


@dataclasses.dataclass(frozen=True)
class Impl(Generic[T]):
    """Implementation of estimation in state-space models."""

    name: str
    rv_from_sqrtnorm: Callable[[jax.Array, jax.Array], T]
    rv_from_mvnorm: Callable[[jax.Array, jax.Array], T]
    rv_to_mvnorm: Callable[[T], tuple[jax.Array, jax.Array]]
    rv_sample: Callable[[Any, T], jax.Array]
    rv_fixedpoint_augment: Callable[[T], T]
    rv_fixedpoint_select: Callable[[T], T]
    rv_logpdf: Callable[[jax.Array, T], jax.Array]
    conditional_parametrize: Callable[[jax.Array, SSMCond[T]], T]
    conditional_merge: Callable[[SSMCond[T], SSMCond[T]], SSMCond[T]]
    conditional_from_identity: Callable[[T], SSMCond[T]]
    dynamics_fixedpoint_augment: Callable[[SSMDynamics[T]], SSMDynamics[T]]
    marginalize: Callable[[T, SSMCond[T]], T]
    bayes_update: Callable[[T, SSMCond[T]], tuple[T, SSMCond[T]]]


def impl_cholesky_based() -> Impl[NormalChol]:
    """Construct a Cholesky-based implementation of estimation in state-space models."""

    def rv_from_mvnorm(m, c) -> NormalChol:
        return NormalChol(m, jnp.linalg.cholesky(c))

    def rv_from_sqrtnorm(m, c) -> NormalChol:
        return NormalChol(m, c)

    def rv_to_mvnorm(rv: NormalChol):
        return rv.mean, rv.cholesky @ rv.cholesky.T

    def marginalize(rv: NormalChol, cond: SSMCond[NormalChol]) -> NormalChol:
        mean = cond.A @ rv.mean + cond.noise.mean

        R_X_F = rv.cholesky.T @ cond.A.T
        R_X = cond.noise.cholesky.T
        cholesky_marg_T = jnp.concatenate([R_X_F, R_X], axis=0)
        cholesky_T = jnp.linalg.qr(cholesky_marg_T, mode="r")
        return NormalChol(mean, cholesky_T.T)

    def bayes_update(rv: NormalChol, cond: SSMCond[NormalChol]):
        R_YX = cond.noise.cholesky.T
        R_X = rv.cholesky.T
        R_X_F = rv.cholesky.T @ cond.A.T
        R_y, (R_xy, G) = _revert_conditional(R_X_F=R_X_F, R_X=R_X, R_YX=R_YX)

        s = cond.A @ rv.mean + cond.noise.mean
        mean_new = rv.mean - G @ s
        return NormalChol(s, R_y.T), SSMCond(G, NormalChol(mean_new, R_xy.T))

    def conditional_parametrize(x: jax.Array, cond: SSMCond[NormalChol]):
        return NormalChol(cond.A @ x + cond.noise.mean, cond.noise.cholesky)

    def rv_sample(key, /, rv: NormalChol) -> jax.Array:
        base = jax.random.normal(key, shape=rv.mean.shape, dtype=rv.mean.dtype)
        return rv.mean + rv.cholesky @ base

    def conditional_merge(
        cond1: SSMCond[NormalChol], cond2: SSMCond[NormalChol]
    ) -> SSMCond[NormalChol]:
        A1, (m1, C1) = cond1.A, cond1.noise
        A2, (m2, C2) = cond2.A, cond2.noise

        A = A1 @ A2
        m = A1 @ m2 + m1
        chol_T = jnp.linalg.qr(jnp.concatenate([C2.T @ A1.T, C1.T]), mode="r")
        return SSMCond(A, NormalChol(m, chol_T.T))

    def conditional_from_identity(like: NormalChol) -> SSMCond[NormalChol]:
        eye = jnp.eye(len(like.mean))
        noise = jax.tree.map(jnp.zeros_like, like)
        return SSMCond(eye, noise)

    def rv_fixedpoint_augment(rv: NormalChol) -> NormalChol:
        mean, chol = rv
        mean_augmented = jnp.concatenate([mean, mean], axis=0)
        chol_augmented = jnp.block(
            [[chol, jnp.zeros_like(chol)], [chol, jnp.zeros_like(chol)]]
        )
        return NormalChol(mean_augmented, chol_augmented)

    def dynamics_fixedpoint_augment(
        dynamics: SSMDynamics[NormalChol],
    ) -> SSMDynamics[NormalChol]:
        # Augment latent dynamics
        A, (q, Q) = dynamics.latent.A, dynamics.latent.noise
        A_ = jax.scipy.linalg.block_diag(A, jnp.eye(len(A)))
        q_ = jnp.concatenate([q, jnp.zeros_like(q)], axis=0)
        Q_ = jax.scipy.linalg.block_diag(Q, jnp.zeros_like(Q))
        latent = SSMCond(A_, NormalChol(q_, Q_))

        # Augment observations
        H, cond = dynamics.observation.A, dynamics.observation.noise
        H_ = jnp.concatenate([H, jnp.zeros_like(H)], axis=1)
        observation = SSMCond(H_, cond)

        # Combine and return
        return SSMDynamics(latent=latent, observation=observation)

    def rv_fixedpoint_select(rv: NormalChol) -> NormalChol:
        mean, cholesky = rv
        n = len(mean) // 2
        mean_new = mean[n:]
        cholesky_new = jnp.linalg.qr(cholesky.T[:, n:], mode="r").T
        return NormalChol(mean_new, cholesky_new)

    def logpdf(u, /, rv):
        # The cholesky factor is triangular, so we compute a cheap slogdet.
        diagonal = jnp.diagonal(rv.cholesky, axis1=-1, axis2=-2)
        slogdet = jnp.sum(jnp.log(jnp.abs(diagonal)))

        dx = u - rv.mean
        residual_white = jax.scipy.linalg.solve_triangular(rv.cholesky.T, dx, trans="T")
        x1 = jnp.dot(residual_white, residual_white)
        x2 = 2.0 * slogdet
        x3 = u.size * jnp.log(jnp.pi * 2)
        return -0.5 * (x1 + x2 + x3)

    return Impl(
        name="Cholesky-based",
        rv_from_mvnorm=rv_from_mvnorm,
        rv_from_sqrtnorm=rv_from_sqrtnorm,
        rv_to_mvnorm=rv_to_mvnorm,
        rv_sample=rv_sample,
        rv_fixedpoint_select=rv_fixedpoint_select,
        rv_fixedpoint_augment=rv_fixedpoint_augment,
        rv_logpdf=logpdf,
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


def impl_covariance_based() -> Impl[NormalCov]:
    """Construct a covariance-based implementation of estimation in state-space models."""

    def rv_sample(key, /, rv: NormalCov) -> jax.Array:
        base = jax.random.normal(key, shape=rv.mean.shape, dtype=rv.mean.dtype)
        return (
            rv.mean + jnp.linalg.cholesky(rv.cov + jnp.eye(len(rv.cov)) * 1e-8) @ base
        )

    def conditional_parametrize(x: jax.Array, /, cond: SSMCond[NormalCov]) -> NormalCov:
        return NormalCov(cond.A @ x + cond.noise.mean, cond.noise.cov)

    def marginalize(rv: NormalCov, /, cond: SSMCond[NormalCov]) -> NormalCov:
        mean = cond.A @ rv.mean + cond.noise.mean
        cov = cond.A @ rv.cov @ cond.A.T + cond.noise.cov
        return NormalCov(mean, cov)

    def bayes_update(
        rv: NormalCov, cond: SSMCond[NormalCov]
    ) -> tuple[NormalCov, SSMCond[NormalCov]]:
        s = cond.A @ rv.mean + cond.noise.mean
        S = cond.A @ rv.cov @ cond.A.T + cond.noise.cov

        # Update via Cholesky decomposition
        s_chol = jax.scipy.linalg.cho_factor(S)
        gain = jax.scipy.linalg.cho_solve(s_chol, cond.A @ rv.cov).T

        mean_new = rv.mean - gain @ s
        cov_new = rv.cov - gain @ S @ gain.T
        return NormalCov(s, S), SSMCond(gain, NormalCov(mean_new, cov_new))

    def rv_fixedpoint_augment(rv: NormalCov) -> NormalCov:
        mean, cov = rv
        mean_augmented = jnp.concatenate([mean, mean], axis=0)
        cov_augmented_row = jnp.concatenate([cov, cov], axis=1)
        cov_augmented = jnp.concatenate([cov_augmented_row, cov_augmented_row], axis=0)
        return NormalCov(mean_augmented, cov_augmented)

    def rv_fixedpoint_select(rv: NormalCov) -> NormalCov:
        mean, cov = rv
        n = len(mean) // 2
        return NormalCov(mean[n:], cov[n:, n:])

    def dynamics_fixedpoint_augment(
        dynamics: SSMDynamics[NormalCov],
    ) -> SSMDynamics[NormalCov]:
        # Augment latent dynamics
        A, (q, Q) = dynamics.latent.A, dynamics.latent.noise
        A_ = jax.scipy.linalg.block_diag(A, jnp.eye(len(A)))
        q_ = jnp.concatenate([q, jnp.zeros_like(q)], axis=0)
        Q_ = jax.scipy.linalg.block_diag(Q, jnp.zeros_like(Q))
        latent = SSMCond(A_, NormalCov(q_, Q_))

        # Augment observations
        H, cond = dynamics.observation.A, dynamics.observation.noise
        H_ = jnp.concatenate([H, jnp.zeros_like(H)], axis=1)
        observation = SSMCond(H_, cond)

        # Combine and return
        return SSMDynamics(latent=latent, observation=observation)

    def conditional_merge(
        cond1: SSMCond[NormalCov], cond2: SSMCond[NormalCov]
    ) -> SSMCond[NormalCov]:
        A1, (m1, C1) = cond1.A, cond1.noise
        A2, (m2, C2) = cond2.A, cond2.noise

        A = A1 @ A2
        m = A1 @ m2 + m1
        C = A1 @ C2 @ A1.T + C1
        return SSMCond(A, NormalCov(m, C))

    def conditional_from_identity(like: NormalCov) -> SSMCond[NormalCov]:
        eye = jnp.eye(len(like.mean))
        noise = jax.tree.map(jnp.zeros_like, like)
        return SSMCond(eye, noise)

    def rv_logpdf(y, rv):
        logpdf = jax.scipy.stats.multivariate_normal.logpdf
        return logpdf(y, rv.mean, rv.cov)

    return Impl(
        name="Covariance-based",
        rv_from_sqrtnorm=lambda m, c: NormalCov(m, c @ c.T),
        rv_to_mvnorm=lambda rv: (rv.mean, rv.cov),
        rv_from_mvnorm=lambda m, c: NormalCov(m, c),
        rv_sample=rv_sample,
        rv_logpdf=rv_logpdf,
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
    dynamics: SSMDynamics[T]


jax.tree_util.register_pytree_node(
    SSM,
    lambda ssm: ((ssm.init, ssm.dynamics), ()),
    lambda a, c: SSM(*c),
)


def ssm_regression_wiener_velocity(ts, /, impl: Impl[T], dim=1) -> Callable:
    """Construct a Wiener-velocity car-tracking model."""

    def parametrize(noise, diffusion=1.0) -> SSM[T]:
        def transition(dt) -> SSMDynamics:
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
            return SSMDynamics(latent=SSMCond(A, rv_q), observation=SSMCond(H, rv_r))

        m0 = jnp.zeros((2 * dim,))
        C0 = jnp.eye(2 * dim)
        x0 = impl.rv_from_mvnorm(m0, C0)

        return SSM(init=x0, dynamics=jax.vmap(transition)(jnp.diff(ts)))

    return parametrize


def ssm_regression_wiener_integrated(ts, /, impl: Impl[T], num: int) -> Callable:
    """Construct an interpolation problem with a seven-derivative Wiener process."""

    def parametrize(noise, diffusion) -> SSM[T]:
        # Get all prior transitions from probdiffeq
        prior = probdiffeq.ivpsolvers.prior_ibm_discrete(
            ts, output_scale=diffusion, num_derivatives=num
        )
        init = prior.init
        inintmean = init.mean.at[0].set(1.0)
        cov = init.cholesky @ init.cholesky.T
        cov = cov.at[0, 0].set(1e-12)
        init = impl.rv_from_mvnorm(inintmean, cov)
        A = prior.conditional.matmul
        q = prior.conditional.noise.mean
        Q = prior.conditional.noise.cholesky
        noise_q = impl.rv_from_mvnorm(q, jax.vmap(lambda s: s @ s.T)(Q))

        # Observations: point-observations of the zeroth state.
        H = jnp.zeros((1, A.shape[1]))
        H = H.at[0, 0].set(1.0)
        r = jnp.zeros((1,))
        R = noise * jnp.eye(1)
        H = jnp.stack([H] * len(ts[:-1]))
        r = jnp.stack([r] * len(ts[:-1]))
        R = jnp.stack([R] * len(ts[:-1]))
        return SSM(
            init=init,
            dynamics=SSMDynamics(
                latent=SSMCond(A, noise_q),
                observation=SSMCond(H, jax.vmap(impl.rv_from_sqrtnorm)(r, R)),
            ),
        )

    return parametrize


def ssm_regression_wiener_acceleration(ts, /, impl: Impl[T], dim: int = 1) -> Callable:
    """Construct a Wiener-acceleration car-tracking model."""

    def parametrize(noise, diffusion=1.0) -> SSM[T]:
        def transition(dt) -> SSMDynamics:
            eye_d = jnp.eye(dim)
            one_d = jnp.ones((dim,))

            A_1d = jnp.asarray([[1.0, dt, dt**2 / 2], [0, 1.0, dt], [0, 0, 1.0]])
            q_1d = jnp.asarray([0.0, 0.0, 0.0])
            Q_1d = diffusion**2 * jnp.asarray(
                [
                    [dt**5 / 20, dt**4 / 8, dt**3 / 3],
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
            return SSMDynamics(latent=SSMCond(A, rv_q), observation=SSMCond(H, rv_r))

        m0 = jnp.zeros((3 * dim,))
        C0 = jnp.eye(3 * dim)
        x0 = impl.rv_from_mvnorm(m0, C0)
        dynamics = jax.vmap(transition)(jnp.diff(ts))
        return SSM(init=x0, dynamics=dynamics)

    return parametrize


def compute_stats_sample(impl: Impl[T]) -> Callable:
    """Sample from a state-space model (sequentially)."""

    def sample(key, ssm: SSM[T]):
        def scan_fun(x, dynamics_k: SSMDynamics):
            key_k, sample_k = x

            key_k, subkey_k = jax.random.split(key_k, num=2)
            rv = impl.conditional_parametrize(sample_k, dynamics_k.latent)
            sample_k = impl.rv_sample(subkey_k, rv)

            key_k, subkey_k = jax.random.split(key_k, num=2)
            rv_obs = impl.conditional_parametrize(sample_k, dynamics_k.observation)
            sample_obs_k = impl.rv_sample(subkey_k, rv_obs)
            return (key_k, sample_k), (sample_k, sample_obs_k)

        key, subkey = jax.random.split(key, num=2)
        x0 = impl.rv_sample(subkey, ssm.init)

        key, subkey = jax.random.split(key, num=2)
        _, (latents, obs) = jax.lax.scan(scan_fun, xs=ssm.dynamics, init=(subkey, x0))
        return jnp.concatenate([x0[None, ...], latents], axis=0), obs

    return sample


def compute_stats_marginalize(impl: Impl[T], reverse: bool) -> Callable:
    """Marginalize a sequence of conditionals (sequentially)."""

    def marginalize(init: T, cond: SSMCond[T]) -> T:
        def scan_fun(x, cond_k: SSMCond[T]):
            marg = impl.marginalize(x, cond_k)
            return marg, marg

        _, all_ = jax.lax.scan(scan_fun, xs=cond, init=init, reverse=reverse)
        if reverse:
            return jax.tree.map(lambda a, b: jnp.concatenate([a, b[None]]), all_, init)
        return jax.tree.map(lambda a, b: jnp.concatenate([a[None], b]), init, all_)

    return marginalize


def compute_fixedpoint(impl: Impl[T], cb: Callable | None = None) -> Callable:
    """Estimate a solution of the fixed-point smoothing problem."""

    class State(NamedTuple):
        rv: T
        cond: SSMCond[T]
        info: dict
        evidence: jax.Array

    def estimate(data: jax.Array, ssm: SSM[T]):
        def step_fun(state_k, inputs: tuple[jax.Array, SSMDynamics]) -> tuple:
            # Read
            (y_k, model_k) = inputs

            # Predict
            state_rv, state_backward = state_k.rv, state_k.cond
            state_kplus, backward = impl.bayes_update(state_rv, model_k.latent)
            backward = impl.conditional_merge(state_backward, backward)

            # Update
            marg, gain = impl.bayes_update(state_kplus, model_k.observation)
            state_new = impl.conditional_parametrize(y_k, gain)

            # Evidence
            ev_local = impl.rv_logpdf(y_k, marg)
            evidence_new = state_k.evidence + ev_local
            info_ = cb(state_new, backward) if cb is not None else {}
            state_updated = State(
                rv=state_new, cond=backward, info=info_, evidence=evidence_new
            )
            return state_updated, {}

        cond = impl.conditional_from_identity(ssm.init)
        info = cb(ssm.init, cond) if cb is not None else {}
        init = State(ssm.init, cond, info=info, evidence=0.0)

        xs = (data, ssm.dynamics)
        final, _ = jax.lax.scan(step_fun, xs=xs, init=init)
        evidence = final.evidence / len(data)

        marginal = impl.marginalize(final.rv, final.cond)
        aux = {"evidence": evidence, **final.info}
        return marginal, aux

    return estimate


def compute_fixedpoint_via_smoother(
    impl: Impl[T], cb: Callable | None = None
) -> Callable:
    """Estimate a solution of the fixed-point smoothing problem.

    Calls a fixed-interval smoother internally.
    """
    compute_interval = compute_fixedinterval(impl=impl, cb=cb)
    marginalize = compute_stats_marginalize(impl=impl, reverse=True)

    def estimate(data: jax.Array, ssm: SSM[T]):
        (terminal, conds), aux = compute_interval(data=data, ssm=ssm)
        marginals = marginalize(terminal, conds)
        initial_rts = jax.tree.map(lambda s: s[0, ...], marginals)
        return initial_rts, aux

    return estimate


def compute_fixedinterval(impl: Impl[T], cb: Callable | None = None) -> Callable:
    """Estimate a solution of the fixed-interval smoothing problem."""

    class State(NamedTuple):
        rv: T
        evidence: jax.Array

    def estimate(data: jax.Array, ssm: SSM[T]):
        def step_fun(state_k: State, inputs: tuple[jax.Array, SSMDynamics]):
            # Read
            (y_k, model_k) = inputs

            # Predict
            state_kplus, cond = impl.bayes_update(state_k.rv, model_k.latent)

            # Update
            marg, gain = impl.bayes_update(state_kplus, model_k.observation)
            state_new = impl.conditional_parametrize(y_k, gain)
            info_ = cb([state_new, cond]) if cb is not None else {}

            # Evidence
            ev_local = impl.rv_logpdf(y_k, marg)
            evidence_new = state_k.evidence + ev_local
            updated = State(state_new, evidence_new)
            return updated, ((state_new, cond), info_)

        init = State(ssm.init, evidence=0.0)

        xs = (data, ssm.dynamics)
        solution, ((filterdists, conds), aux) = jax.lax.scan(step_fun, xs=xs, init=init)
        evidence = solution.evidence / len(data)

        return (solution.rv, conds), {
            "evidence": evidence,
            "filter_distributions": filterdists,
            **aux,
        }

    return estimate


def compute_fixedpoint_via_filter(
    impl: Impl[T], cb: Callable | None = None
) -> Callable:
    """Estimate a solution of the fixed-point smoothing problem.

    Augments the state-space model and calls a filter internally.
    """
    compute_fi = compute_filter(impl=impl, cb=cb)

    def estimate(data: jax.Array, ssm: SSM[T]):
        ssm_augment = _ssm_augment_fixedpoint(ssm, impl=impl)
        terminal_filter, aux = compute_fi(data, ssm_augment)
        rv_reduced = impl.rv_fixedpoint_select(terminal_filter)
        return rv_reduced, {"evidence": 0.0, **aux}

    return estimate


def _ssm_augment_fixedpoint(ssm: SSM[T], impl: Impl[T]) -> SSM[T]:
    init = impl.rv_fixedpoint_augment(ssm.init)
    dynamics = jax.vmap(impl.dynamics_fixedpoint_augment)(ssm.dynamics)
    return SSM(init=init, dynamics=dynamics)


def compute_filter(impl: Impl[T], cb: Callable | None = None) -> Callable:
    """Estimate a solution of the filtering problem."""

    class State(NamedTuple):
        rv: T
        info: dict
        evidence: jax.Array

    def estimate(data: jax.Array, ssm: SSM[T]):
        def step_fun(state_k: State, inputs: tuple[jax.Array, SSMDynamics]):
            # Read
            (y_k, model_k) = inputs

            # Predict
            state_kplus = impl.marginalize(state_k.rv, model_k.latent)

            # Update
            marg, cond = impl.bayes_update(state_kplus, model_k.observation)
            state_new = impl.conditional_parametrize(y_k, cond)

            # Evidence
            evidence_local = impl.rv_logpdf(y_k, marg)

            info_k = cb(state_new) if cb is not None else {}
            evidence = state_k.evidence + evidence_local
            return State(rv=state_new, info=info_k, evidence=evidence), {}

        x0 = ssm.init
        info = cb(x0) if cb is not None else {}
        init = State(rv=ssm.init, info=info, evidence=0.0)

        final, _ = jax.lax.scan(step_fun, xs=(data, ssm.dynamics), init=init)
        evidence = final.evidence / len(data)
        return final.rv, {"evidence": evidence, **final.info}

    return estimate
