"""State-space models and their parametrisations."""

import dataclasses
import jax.numpy as jnp
import jax


@dataclasses.dataclass
class Parametrisation:
    pass


def param_conventional():
    return Parametrisation()


def param_square_root():
    return Parametrisation()


def model_car_tracking_velocity(ts, /, noise, diffusion, *, param):
    def transition(dt):
        eye_d = jnp.eye(2)

        A_1d = jnp.asarray([[1.0, dt], [0, 1.0]])
        q_1d = jnp.asarray([0.0, 0.0])
        Q_1d = diffusion * jnp.asarray([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
        H_1d = jnp.asarray([1.0, 0.0])
        r_1d = jnp.asarray(0.0)
        R_1d = noise * jnp.asarray(1.0)

        A = jnp.kron(A_1d, eye_d)
        q = jnp.kron(q_1d, eye_d)
        Q = jnp.kron(Q_1d, eye_d)
        H = jnp.kron(H_1d, eye_d)
        r = jnp.kron(r_1d, eye_d)
        R = jnp.kron(R_1d, eye_d)

        rv_q = param.rv_from_multivariate_normal(q, Q)
        rv_r = param.rv_from_multivariate_normal(r, R)
        return (A, rv_q), (H, rv_r)

    return jax.vmap(transition)(jnp.diff(ts))


def sample(key, model, *, param):
    pass
