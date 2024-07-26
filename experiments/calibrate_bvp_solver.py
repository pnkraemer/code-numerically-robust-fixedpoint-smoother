import jax.flatten_util
import jax.numpy as jnp
from fpx import fpx


def main():
    t0 = 1.0 / (jnp.pi * 3)
    t1 = 1.0
    y0 = 0.0
    y1 = jnp.sin(1.0)

    ts = jnp.linspace(t0, t1, endpoint=True, num=10)
    impl = fpx.impl_cholesky_based()
    constraint = information_operator(ts, impl=impl)
    print(jax.tree.map(jnp.shape, constraint))


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
        h = res_wrapped(t, x_flat)
        noise = impl.rv_from_mvnorm(jnp.zeros_like(h), jnp.eye(1))
        return fpx.SSMCond(H[None], noise=noise)

    return jax.vmap(linearized)(ts)


if __name__ == "__main__":
    main()
