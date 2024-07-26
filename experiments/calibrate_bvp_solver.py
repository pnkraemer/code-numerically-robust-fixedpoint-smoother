import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tqdm
from fpx import fpx


def main():
    jax.config.update("jax_enable_x64", True)

    num_iterations = 3

    # Select a BVP
    t0 = 1.0 / (jnp.pi * 3)
    t1 = 1.0
    y0 = jnp.atleast_1d(0.0)
    y1 = jnp.atleast_1d(jnp.sin(1.0))

    def vector_field(t, *xs):
        # Same as in matlab:
        # https://www.mathworks.com/help/matlab/ref/bvp5c.html
        return xs[2] + 2 * xs[1] / t + xs[0] / t**4

    # Build a state-space model
    num = 7
    impl = fpx.impl_cholesky_based()
    ts = jnp.linspace(t0, t1, endpoint=True, num=50)
    init = model_init(impl=impl, num=num)
    latent = model_latent(ts, impl=impl, num=num)
    constraint = model_constraint(ts, vf=vector_field, impl=impl, num=num)

    # Replace the final constraint with the BCond
    constraint = model_constraint_replace_y1(y1, cond=constraint, impl=impl, num=num)

    # We handle the initial constraint separately, so it must be
    # split from the remaining constraints
    constraint_t0 = jax.tree.map(lambda s: s[0, ...], constraint)
    constraint_ts = jax.tree.map(lambda s: s[1:, ...], constraint)

    # Update the initial condition on y0
    interp_t0 = model_interpolation(impl=impl, num=num)
    _, cond_interp_t0 = impl.bayes_update(init, interp_t0)
    init = impl.conditional_parametrize(y0, cond_interp_t0)

    # Update the initial condition on the ODE constraint
    zeros = jnp.zeros((1,))
    _, cond_constraint_t0 = impl.bayes_update(init, constraint_t0)
    init = impl.conditional_parametrize(zeros, cond_constraint_t0)

    # Construct the state-space model including
    # initial condition, constraints, and latent dynamics
    dynamics = fpx.SSMDynamics(latent, constraint_ts)
    ssm = fpx.SSM(init=init, dynamics=dynamics)
    data = jnp.zeros((len(ts[1:]), 1))

    # Construct a fixed-point smoother
    fp_smoother = fpx.compute_fixedpoint(impl=impl)

    # Run a few fixed-point smoother iterations
    progressbar = tqdm.tqdm(range(num_iterations))
    delta = jnp.inf
    for _ in progressbar:
        progressbar.set_description(f"Delta: {delta:.1e}")
        init_estimated, _info = fp_smoother(data=data, ssm=ssm)
        init, delta = em_update_init(init, init_estimated, impl=impl)

    print("Final estimate:", init.mean)
    # Construct a fixed-interval smoother so we can plot the solution
    fi_smoother = fpx.compute_fixedinterval(impl=impl)
    (final, conds), _info = fi_smoother(data=data, ssm=ssm)

    marginalize = fpx.compute_stats_marginalize(impl=impl, reverse=True)
    marginals = marginalize(final, conds)

    # Plot the results
    plt.plot(ts, marginals.mean[:, 0], label="Approximation")
    plt.plot(ts, jnp.sin(1 / ts), label="Truth")
    plt.legend()
    plt.show()

    # Plot the residuals
    error = jax.vmap(vector_field)(ts, *marginals.mean.T)
    plt.semilogy(ts, jnp.abs(error), label="Residual")
    plt.legend()
    plt.show()


def model_init(*, impl, num):
    mean = jnp.ones((num + 1,))
    cholesky = jnp.eye(num + 1)
    return impl.rv_from_sqrtnorm(mean, cholesky)


def model_latent(ts, *, impl, num) -> fpx.SSMCond:
    ssm = fpx.ssm_wiener_integrated_interpolation(ts, impl=impl, num=num)
    return ssm.dynamics.latent


def model_constraint(ts, *, vf, impl, num) -> fpx.SSMCond:
    # todo: if we also vectorise over x_flat and pass this as an input,
    #  we automatically handle nonlinear problems
    x_and_dx = [1.0] * (num + 1)
    x_flat, unflatten = jax.flatten_util.ravel_pytree(x_and_dx)

    def vf_wrapped(t, xflat):
        return vf(t, *unflatten(xflat))

    def linearized(t) -> fpx.SSMCond:
        H = jax.jacfwd(vf_wrapped, argnums=1)(t, x_flat)
        h = vf_wrapped(t, x_flat)[None]
        h = h - H @ x_flat
        noise = impl.rv_from_sqrtnorm(h, 0 * jnp.eye(1))
        return fpx.SSMCond(H[None], noise=noise)

    return jax.vmap(linearized)(ts)


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


def em_update_init(init, estimated, impl):
    # Update the mean
    mean_new = estimated.mean

    # Update the Cholesky factor
    diff = estimated.mean - init.mean
    stack = jnp.concatenate([init.cholesky.T, diff[None, ...]], axis=0)
    cholesky_new = jnp.linalg.qr(stack, mode="r").T
    updated = impl.rv_from_sqrtnorm(mean_new, cholesky_new)

    # Compute the difference between updates
    flat_old, _ = jax.flatten_util.ravel_pytree(impl.rv_to_mvnorm(init))
    flat_new, _ = jax.flatten_util.ravel_pytree(impl.rv_to_mvnorm(updated))
    delta = jnp.linalg.norm(
        (flat_old - flat_new) / (1e-10 + jnp.abs(flat_old))
    ) / jnp.sqrt(flat_old.size)
    return updated, delta


if __name__ == "__main__":
    main()
