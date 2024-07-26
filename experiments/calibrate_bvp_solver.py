import jax.flatten_util
import jax.numpy as jnp


def res(t, x, dx, ddx):
    return ddx + 2 * dx / t + x / t**4


x_and_dx = [1.0, 1.0, 10.0]
x_flat, unflatten = jax.flatten_util.ravel_pytree(x_and_dx)


def res_wrapped(t, xflat):
    return res(t, *unflatten(xflat))


def Ht(t):
    return jax.jacfwd(res_wrapped, argnums=1)(t, x_flat)


print(Ht(2.0))
