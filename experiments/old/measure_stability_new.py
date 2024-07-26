import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tqdm
from fpx import fpx

# todo: track marginal likelihoods of the data and compare for different means
# todo: use car tracking for parameter estimation
# todo: use autoregressive model for complexity estimates
# todo: use BVP for stability


def main():
    num_iterations = 20

    # Choose an implementation
    impl = fpx.impl_cholesky_based()

    # Construct a fixed-point smoother
    fp_smoother = fpx.compute_fixedpoint(impl=impl)
    init = simulate_bvp(impl, fp_smoother, num_steps=100)
    print(init)
    assert False


def simulate_bvp(impl, fp_smoother, num_steps):
    # Select a BVP
    # vector_field, (t0, t1), (y0, y1), solution = bvp_matlab()
    vector_field, (t0, t1), (y0, y1), solution = bvp_linear_15th(scale=1e-3)

    # Build a state-space model
    ts = jnp.linspace(t0, t1, endpoint=True, num=num_steps)
    num = 2
    init = model_init(impl=impl, num=num)
    latent = model_latent(ts, impl=impl, num=num)

    # Linearize (if already linear, this step evaluates system matrices)
    x_and_dx = [1.0] * (num + 1)
    x_flat, unflatten = jax.flatten_util.ravel_pytree(x_and_dx)
    xs = jnp.stack([x_flat] * len(ts[1:]), axis=0)
    constraint = model_constraint(
        ts[1:], xs, vector_field, unflatten=unflatten, impl=impl
    )

    # Replace the final constraint with the BCond
    constraint = model_constraint_replace_y1(y1, cond=constraint, impl=impl, num=num)

    # Update the initial condition on y0
    interp_t0 = model_interpolation(impl=impl, num=num)
    _, cond_interp_t0 = impl.bayes_update(init, interp_t0)
    init = impl.conditional_parametrize(y0, cond_interp_t0)

    # Construct the state-space model including
    # initial condition, constraints, and latent dynamics
    dynamics = fpx.SSMDynamics(latent, constraint)
    data = jnp.zeros((len(ts[1:]), 1))
    ssm = fpx.SSM(init=init, dynamics=dynamics)

    # Run a few fixed-point smoother iterations
    init_estimated, _info = fp_smoother(data=data, ssm=ssm)
    return init_estimated


def other():
    # Construct a fixed-interval smoother so we can plot the solution
    fi_smoother = fpx.compute_fixedinterval(impl=impl)
    (final, conds), _info = fi_smoother(data=data, ssm=ssm)

    marginalize = fpx.compute_stats_marginalize(impl=impl, reverse=True)
    marginals = marginalize(final, conds)

    # Plot the results
    plt.plot(ts, marginals.mean[:, 0], label="Approximation")
    if solution is not None:
        print(jnp.linalg.norm(jax.vmap(solution)(ts) - marginals.mean[:, 0]))
        plt.plot(ts, jax.vmap(solution)(ts), label="Truth")
    plt.legend()
    plt.show()

    # Plot the residuals
    error = jax.vmap(vector_field)(ts, *marginals.mean.T)
    plt.semilogy(ts, jnp.abs(error), label="Residual")
    plt.legend()
    plt.show()


def bvp_matlab():
    t0 = 1.0 / (jnp.pi * 3)
    t1 = 1.0
    y0 = jnp.atleast_1d(0.0)
    y1 = jnp.atleast_1d(jnp.sin(1.0))

    def vector_field(t, *xs):
        # Same as in matlab:
        # https://www.mathworks.com/help/matlab/ref/bvp5c.html
        return xs[2] + 2 * xs[1] / t + xs[0] / t**4

    def solution(t):
        return jnp.sin(1 / t)

    return vector_field, (t0, t1), (y0, y1), solution


def bvp_linear_15th(scale=0.1):
    t0 = -1.0
    t1 = 1.0
    y0 = 1.0
    y1 = 1.0

    def vector_field(t, *xs):
        return scale * xs[2] - xs[0] * t

    y0 = jnp.atleast_1d(y0)
    y1 = jnp.atleast_1d(y1)
    return vector_field, (t0, t1), (y0, y1), None


def bvp_nonlinear_20th(scale=0.1):
    t0 = 0.0
    t1 = 1.0
    y0 = 1 + scale * jnp.log(jnp.cosh(-0.745 / scale))
    y1 = 1 + scale * jnp.log(jnp.cosh(0.255 / scale))

    def vector_field(_t, *xs):
        return scale * xs[2] + xs[1] ** 2 - 1.0

    def solution(t):
        return 1 + scale * jnp.log(jnp.cosh(t - 0.745) / scale)

    y0 = jnp.atleast_1d(y0)
    y1 = jnp.atleast_1d(y1)
    return vector_field, (t0, t1), (y0, y1), solution


def model_init(*, impl, num):
    mean = jnp.ones((num + 1,))
    cholesky = 10_000 * jnp.eye(num + 1)  # diffuse initialisation
    return impl.rv_from_sqrtnorm(mean, cholesky)


def model_latent(ts, *, impl, num) -> fpx.SSMCond:
    ssm = fpx.ssm_wiener_integrated_interpolation(ts, impl=impl, num=num)
    return ssm.dynamics.latent


def model_constraint(ts, xs, vf, *, unflatten, impl) -> fpx.SSMCond:
    def vf_wrapped(t, xflat):
        return vf(t, *unflatten(xflat))

    def linearized(t, x_flat) -> fpx.SSMCond:
        H = jax.jacfwd(vf_wrapped, argnums=1)(t, x_flat)
        h = vf_wrapped(t, x_flat)[None]
        h = h - H @ x_flat
        noise = impl.rv_from_sqrtnorm(h, 0.0 * jnp.eye(1))
        return fpx.SSMCond(H[None], noise=noise)

    return jax.vmap(linearized)(ts, xs)


def model_constraint_replace_y1(y1, cond: fpx.SSMCond, impl, num) -> fpx.SSMCond:
    A = cond.A
    noise_mean = cond.noise.mean

    A0 = jnp.eye(num + 1)[[0], :]
    # A = A.at[0, ...].set(A0)
    A = A.at[-1, ...].set(A0)

    # b0 = jnp.atleast_1d(y0)
    b1 = -jnp.atleast_1d(y1)
    # noise_mean = noise_mean.at[0, ...].set(b0)
    noise_mean = noise_mean.at[-1, ...].set(b1)
    noise = impl.rv_from_sqrtnorm(noise_mean, cond.noise.cholesky)

    return fpx.SSMCond(A, noise)


def model_interpolation(*, impl, num):
    A = jnp.eye(num + 1)[[0], ...]
    r = jnp.zeros((1,))
    R = jnp.zeros((1, 1))
    return fpx.SSMCond(A, noise=impl.rv_from_sqrtnorm(r, R))


def em_update_init(*, old, new, impl):
    # Update the mean
    mean_new = new.mean

    # Update the Cholesky factor
    diff = new.mean - old.mean
    stack = jnp.concatenate([new.cholesky.T, diff[None, ...]], axis=0)
    cholesky_new = jnp.linalg.qr(stack, mode="r").T
    updated = impl.rv_from_sqrtnorm(mean_new, cholesky_new)

    # Compute the difference between updates
    flat_old, _ = jax.flatten_util.ravel_pytree(impl.rv_to_mvnorm(old))
    flat_new, _ = jax.flatten_util.ravel_pytree(impl.rv_to_mvnorm(updated))
    delta_abs = jnp.abs(flat_old - flat_new)

    nugget = jnp.sqrt(jnp.finfo(flat_old.dtype).eps)
    delta_rel = delta_abs / (nugget + jnp.abs(flat_old))
    return updated, jnp.linalg.norm(delta_rel) / jnp.sqrt(flat_old.size)


if __name__ == "__main__":
    main()
