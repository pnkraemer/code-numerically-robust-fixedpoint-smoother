import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tqdm
from fpx import fpx


def main():
    jax.config.update("jax_enable_x64", True)

    t0 = 1.0 / (jnp.pi * 3)
    t1 = 1.0
    y0 = 0.0
    y1 = jnp.sin(1.0)

    impl = fpx.impl_cholesky_based()
    ts = jnp.linspace(t0, t1, endpoint=True, num=1000)

    init, prior = prior_transitions(ts, impl=impl)
    # init_mean = init.mean
    init = impl.rv_from_sqrtnorm(10 * jnp.ones_like(init.mean), init.cholesky)
    constraint = information_operator(ts, impl=impl)
    # constraint = replace_bcond_t1(y1, impl=impl, cond=constraint)
    initconstraint = jax.tree.map(lambda s: s[0, ...], constraint)
    constraint = jax.tree.map(lambda s: s[1:, ...], constraint)

    # Update init
    A = jnp.eye(6)[[0], ...]
    r = jnp.zeros((1,))
    R = jnp.zeros((1, 1))
    cond = fpx.SSMCond(A, noise=impl.rv_from_sqrtnorm(r, R))
    _, cond_rev = impl.bayes_update(init, cond)
    init_new = impl.conditional_parametrize(jnp.atleast_1d(y0), cond_rev)
    _, cond_rev = impl.bayes_update(init_new, initconstraint)
    init_new = impl.conditional_parametrize(jnp.zeros_like(r), cond_rev)

    for _ in tqdm.tqdm(range(10)):
        # Final SSM:
        ssm = fpx.SSM(init=init_new, dynamics=fpx.SSMDynamics(prior, constraint))
        # Run an RTS smoother and plot the solution
        rts = fpx.compute_fixedinterval(impl)
        data = jnp.zeros((len(ts[1:]), 1))

        (final, conds), aux = rts(data, ssm)

        A = jnp.eye(6)[[0], ...]
        r = jnp.zeros((1,))
        R = jnp.zeros((1, 1))
        cond = fpx.SSMCond(A, noise=impl.rv_from_sqrtnorm(r, R))
        _, cond_rev = impl.bayes_update(final, cond)
        final = impl.conditional_parametrize(jnp.atleast_1d(y1), cond_rev)

        marginalize = fpx.compute_stats_marginalize(impl=impl, reverse=True)
        marginals = marginalize(final, conds)

        mean_new = marginals.mean[0, :]
        stack = jnp.concatenate(
            [marginals.cholesky[0].T, (mean_new - init_new.mean)[None, ...]], axis=0
        )
        cholesky_new = jnp.linalg.qr(stack, mode="r").T
        init_new = impl.rv_from_sqrtnorm(mean_new, cholesky_new)

    # marginals = aux["filter_distributions"]
    plt.plot(ts, marginals.mean[:, 0], label="Approximation")
    plt.plot(ts, jnp.sin(1 / ts), label="Truth")
    plt.legend()
    plt.show()

    error = jax.vmap(res)(ts, *(marginals.mean.T))
    plt.semilogy(ts, jnp.abs(error), label="Approximation")
    plt.show()


def res(t, x, dx, ddx, d3x, d4x, d5x):
    # Same as in matlab:
    # https://www.mathworks.com/help/matlab/ref/bvp5c.html
    return ddx + 2 * dx / t + x / t**4


def information_operator(ts, *, impl) -> fpx.SSMCond:
    x_and_dx = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    x_flat, unflatten = jax.flatten_util.ravel_pytree(x_and_dx)

    def res_wrapped(t, xflat):
        return res(t, *unflatten(xflat))

    def linearized(t) -> fpx.SSMCond:
        H = jax.jacfwd(res_wrapped, argnums=1)(t, x_flat)
        h = res_wrapped(t, x_flat)[None]
        h = h - H @ x_flat
        noise = impl.rv_from_sqrtnorm(h, 0 * jnp.eye(1))
        return fpx.SSMCond(H[None], noise=noise)

    return jax.vmap(linearized)(ts)


def replace_bcond_t1(y1, impl, cond: fpx.SSMCond) -> fpx.SSMCond:
    A = cond.A
    noise_mean = cond.noise.mean

    A0 = jnp.eye(6)[[0], :]
    # A = A.at[0, ...].set(A0)
    A = A.at[-1, ...].set(A0)

    # b0 = jnp.atleast_1d(y0)
    b1 = -jnp.atleast_1d(y1)
    # noise_mean = noise_mean.at[0, ...].set(b0)
    noise_mean = noise_mean.at[-1, ...].set(b1)
    noise = impl.rv_from_sqrtnorm(noise_mean, cond.noise.cholesky)

    return fpx.SSMCond(A, noise)


def prior_transitions(ts, *, impl):
    # ssm = fpx.ssm_car_tracking_acceleration(ts, impl=impl)
    ssm = fpx.ssm_wiener_integrated_interpolation(ts, impl=impl, num=5)
    return ssm.init, ssm.dynamics.latent


if __name__ == "__main__":
    main()
