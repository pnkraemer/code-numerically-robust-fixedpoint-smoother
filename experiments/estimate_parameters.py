import os

import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tqdm
from fpx import fpx
from tueplots import axes, fonts

plt.rcParams.update(axes.lines())
plt.rcParams.update(axes.tick_direction(y="in", x="in"))
plt.rcParams.update(fonts.jmlr2001_tex())


def main(seed=3):
    jax.config.update("jax_enable_x64", True)
    num_iterations = 3
    key = jax.random.PRNGKey(seed)
    impl = fpx.impl_cholesky_based()

    # Build a state-space model
    ts = jnp.linspace(0.0, 1.0, endpoint=True, num=10)
    ssm = fpx.ssm_car_tracking_velocity(ts, noise=0.1, dim=2, impl=impl)
    init, dynamics = ssm.init, ssm.dynamics
    key, subkey = jax.random.split(key, num=2)
    mean = jax.random.normal(subkey, shape=init.mean.shape)
    key, subkey = jax.random.split(key, num=2)
    cholesky = jax.random.normal(subkey, shape=init.cholesky.shape)
    init = impl.rv_from_sqrtnorm(mean, cholesky)
    ssm = fpx.SSM(init, dynamics)
    ssm_true = ssm  # save the ssm as the correct one

    # Sample data from the correct state-space model
    sample = fpx.compute_stats_sample(impl=impl)
    key, subkey = jax.random.split(key, num=2)
    latent, data = sample(subkey, ssm)

    # Build a fixed-point smoother
    fp_smoother = fpx.compute_fixedpoint(impl=impl)

    # Run the fixed-point smoother in the right model
    init_estimated_true, info = fp_smoother(data=data, ssm=ssm_true)
    print(init_estimated_true)

    # Build a state-space model with the wrong initial condition (wrong mean)
    key, subkey = jax.random.split(key, num=2)
    mean = 10 * jax.random.normal(subkey, shape=init.mean.shape)
    init = impl.rv_from_sqrtnorm(mean, init.cholesky)
    ssm = fpx.SSM(init, dynamics)

    # Create a big figure (to be filled)
    fig, axes = plt.subplots(
        nrows=2,
        ncols=num_iterations,
        sharex=False,
        sharey=True,
        figsize=(8, 8 / num_iterations),
        constrained_layout=True,
    )
    axes[0, 0].set_ylabel(r"PDF ($\theta_1$)")
    axes[1, 0].set_ylabel(r"PDF ($\theta_2$)")

    # Run expectation maximisation around fixed-point smoother iterations
    # and plot the evolution of the PDFs
    for i, axes_i in zip(range(num_iterations), axes.T):
        # Run fixedpoint smoother and carry out EM update
        init_estimated, info = fp_smoother(data=data, ssm=ssm)
        init, delta = em_update_init(old=init, new=init_estimated, impl=impl)
        ssm = fpx.SSM(init=init, dynamics=dynamics)

        # Plot the first coordinate
        xs = jnp.linspace(data[0, 0] - 1 / 6, data[0, 0] + 1 / 3, num=100)
        x0 = init_estimated  # alias to avoid linebreak in the next line
        plot_pdf(axes_i[0], xs, x0, i=0, impl=impl, color="C0", label="Iterate")
        t0 = init_estimated_true  # alias to avoid linebreak in the next line
        plot_pdf(axes_i[0], xs, t0, i=0, impl=impl, color="C1", label="Target")
        axes_i[0].set_title(f"Evidence: {info['likelihood']:.2f}", fontsize="medium")
        axes_i[0].axvline(data[0, 0], label="Noisy data", color="black")
        axes_i[0].legend(fontsize="x-small")
        axes_i[0].set_xlim((jnp.amin(xs), jnp.amax(xs)))
        axes_i[0].set_xlabel("Realisation")

        # Plot the second coordinate
        xs = jnp.linspace(data[0, 1] - 1 / 3, data[0, 1] + 1 / 6, num=100)
        x0 = init_estimated  # alias to avoid linebreak in the next line
        plot_pdf(axes_i[1], xs, x0, i=1, impl=impl, color="C0", label="Iterate")
        t0 = init_estimated_true  # alias to avoid linebreak in the next line
        plot_pdf(axes_i[1], xs, t0, i=1, impl=impl, color="C1", label="Target")
        axes_i[1].axvline(data[0, 1], label="Noisy data", color="black")
        axes_i[1].legend(fontsize="x-small")
        axes_i[1].set_xlim((jnp.amin(xs), jnp.amax(xs)))
        axes_i[1].set_xlabel("Realisation")

    name = os.path.basename(__file__)
    name = name.replace(".py", "")
    plt.savefig(f"./from_results_to_paper/{name}.pdf")
    plt.show()


def plot_pdf(ax, xs, rv, *, i, label, color, impl):
    @jax.vmap
    def pdf(x):
        m, c = impl.rv_to_mvnorm(rv)
        return jax.scipy.stats.norm.pdf(x, m[i], c[i, i])

    ax.plot(xs, pdf(xs), label=label, color=color)
    ax.fill_between(xs, 0, pdf(xs), color=color, alpha=0.15)


def em_update_init(*, old, new, impl):
    # Update the mean
    mean_new = new.mean

    # Update the Cholesky factor
    diff = new.mean - old.mean
    stack = jnp.concatenate([new.cholesky.T, diff[None, ...]], axis=0)
    cholesky_new = jnp.linalg.qr(stack, mode="r").T
    # updated = impl.rv_from_sqrtnorm(old.mean, cholesky_new)
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
