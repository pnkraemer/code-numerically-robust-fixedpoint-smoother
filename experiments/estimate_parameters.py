import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tqdm
from fpx import fpx


def main():
    jax.config.update("jax_enable_x64", True)
    num_iterations = 10

    # Build a state-space model and sample data
    key = jax.random.PRNGKey(1)
    ts = jnp.linspace(0.0, 1.0, endpoint=True, num=10)
    impl = fpx.impl_cholesky_based()
    ssm = fpx.ssm_car_tracking_velocity(ts, noise=0.001, dim=2, impl=impl)
    init, dynamics = ssm.init, ssm.dynamics
    key, subkey = jax.random.split(key, num=2)
    mean = jax.random.normal(subkey, shape=init.mean.shape)
    key, subkey = jax.random.split(key, num=2)
    cholesky = jax.random.normal(subkey, shape=init.cholesky.shape)
    # cholesky = jnp.eye(len(init.cholesky)) * 1e-4
    init = impl.rv_from_sqrtnorm(mean, cholesky)
    ssm = fpx.SSM(init, dynamics)
    sample = fpx.compute_stats_sample(impl=impl)
    key, subkey = jax.random.split(key, num=2)
    latent, data = sample(subkey, ssm)
    ssm_true = ssm

    # Build a fixed-point smoother
    fp_smoother = fpx.compute_fixedpoint(impl=impl)

    # Run the fixed-point smoother in the right model
    init_estimated_true, info = fp_smoother(data=data, ssm=ssm_true)

    # Build a state-space model with the wrong initial condition
    key, subkey = jax.random.split(key, num=2)
    mean = jax.random.normal(subkey, shape=init.mean.shape)
    # key, subkey = jax.random.split(key, num=2)
    # cholesky = jax.random.normal(subkey, shape=init.cholesky.shape)
    init = impl.rv_from_sqrtnorm(mean, init.cholesky)
    ssm = fpx.SSM(init, dynamics)

    # Create a big plot
    fig, axes = plt.subplots(nrows=2, ncols=num_iterations, sharey=True)

    # Run a few fixed-point smoothe iterations
    for i, axes_row in zip(range(num_iterations), axes.T):
        init_estimated, info = fp_smoother(data=data, ssm=ssm)
        init, delta = em_update_init(old=init, new=init_estimated, impl=impl)
        ssm = fpx.SSM(init=init, dynamics=dynamics)

        def pdf(x):
            m, c = impl.rv_to_mvnorm(init_estimated)
            # return -jnp.log(c[0, 0]) - 0.5 + (x - m[0])**2 / c[0, 0]**2 - 0.5*jnp.log(2*jnp.pi)
            return jax.scipy.stats.norm.logpdf(x, m[0], c[0, 0])

        def pdf_true(x):
            m_, c_ = impl.rv_to_mvnorm(init_estimated_true)
            # return -jnp.log(c[0, 0]) - 0.5 + (x - m[0])**2 / c[0, 0]**2 - 0.5*jnp.log(2*jnp.pi)
            return jax.scipy.stats.norm.logpdf(x, m_[0], c_[0, 0])

        xs = jnp.linspace(latent[0, 0] - 0.15, latent[0, 0] + 0.15)
        axes_row[0].plot(xs, jax.vmap(pdf)(xs), label="Estimate")
        axes_row[0].plot(xs, jax.vmap(pdf_true)(xs), label="Truth")
        axes_row[0].set_title(f"Likelihood: {info['likelihood']:.1f}")

        axes_row[0].axvline(latent[0, 0], color="black")
        axes_row[0].legend()
        # axes_row[0].set_ylim((-10, 10.))
        print(i, info["likelihood"])

    plt.show()

    print(init_estimated.mean)
    print(init_estimated_true.mean)
    print()
    print(init_estimated.cholesky @ init_estimated.cholesky.T)
    print(init_estimated_true.cholesky @ init_estimated_true.cholesky.T)


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
    # updated = impl.rv_from_sqrtnorm(mean_new, cholesky_new)
    updated = impl.rv_from_sqrtnorm(mean_new, old.cholesky)

    # Compute the difference between updates
    flat_old, _ = jax.flatten_util.ravel_pytree(impl.rv_to_mvnorm(old))
    flat_new, _ = jax.flatten_util.ravel_pytree(impl.rv_to_mvnorm(updated))
    delta_abs = jnp.abs(flat_old - flat_new)

    nugget = jnp.sqrt(jnp.finfo(flat_old.dtype).eps)
    delta_rel = delta_abs / (nugget + jnp.abs(flat_old))
    return updated, jnp.linalg.norm(delta_rel) / jnp.sqrt(flat_old.size)


if __name__ == "__main__":
    main()
