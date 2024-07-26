import jax.flatten_util
import jax.numpy as jnp
from fpx import fpx


def main():
    t0 = 1.0 / (jnp.pi * 3)
    t1 = 1.0
    y0 = 0.0
    y1 = jnp.sin(1.0)

    impl = fpx.impl_cholesky_based()
    ts = jnp.linspace(t0, t1, endpoint=True, num=10)

    init, prior = prior_transitions(ts, impl=impl)
    constraint = information_operator(ts[1:], impl=impl)
    constraint = replace_bcond_t1(y1, impl=impl, cond=constraint)

    # Update init
    A = jnp.eye(3)[[0], ...]
    r = jnp.zeros((1, 3))
    R = jnp.zeros((1, 1))
    cond = fpx.SSMCond(A, noise=impl.rv_from_sqrtnorm(r, R))
    _, cond_rev = impl.bayes_update(init, cond)
    init_new = impl.conditional_parametrize(jnp.atleast_1d(y0), cond_rev)

    # Final SSM:
    ssm = fpx.SSM(init=init_new, dynamics=fpx.SSMDynamics(prior, constraint))


def information_operator(ts, *, impl) -> fpx.SSMCond:
    def res(t, x, dx, ddx):
        # Same as in matlab:
        # https://www.mathworks.com/help/matlab/ref/bvp5c.html
        return ddx + 2 * dx / t + x / t**4

    x_and_dx = [1.0, 1.0, 10.0]
    x_flat, unflatten = jax.flatten_util.ravel_pytree(x_and_dx)

    def res_wrapped(t, xflat):
        return res(t, *unflatten(xflat))

    def linearized(t) -> fpx.SSMCond:
        H = jax.jacfwd(res_wrapped, argnums=1)(t, x_flat)
        h = res_wrapped(t, x_flat)[None]
        noise = impl.rv_from_sqrtnorm(jnp.zeros_like(h), 0 * jnp.eye(1))
        return fpx.SSMCond(H[None], noise=noise)

    return jax.vmap(linearized)(ts)


def replace_bcond_t1(y1, impl, cond: fpx.SSMCond) -> fpx.SSMCond:
    A = cond.A
    noise_mean = cond.noise.mean

    A0 = jnp.eye(3)[[0], :]
    # A = A.at[0, ...].set(A0)
    A = A.at[-1, ...].set(A0)

    # b0 = jnp.atleast_1d(y0)
    b1 = jnp.atleast_1d(y1)
    # noise_mean = noise_mean.at[0, ...].set(b0)
    noise_mean = noise_mean.at[-1, ...].set(b1)
    noise = impl.rv_from_sqrtnorm(noise_mean, cond.noise.cholesky)

    return fpx.SSMCond(A, noise)


def prior_transitions(ts, *, impl):
    ssm = fpx.ssm_car_tracking_acceleration(ts, impl=impl)
    return ssm.init, ssm.dynamics.latent


if __name__ == "__main__":
    main()
