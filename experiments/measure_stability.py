import jax.numpy as jnpimport jaximport osfrom fpx import fpximport argparseimport pickle# jax.config.update("jax_enable_x64", True)def main():    args = parse_args()    print(args)    results = run_experiment(args)    dirname = str(__file__)    dirname = dirname.replace("experiments", "results")    dirname = dirname.replace(".py", "")    os.makedirs(dirname, exist_ok=True)    filename = f"{dirname}/results.pkl"    with open(filename, "wb") as f:        pickle.dump(results, f)    print(f"Saved results to {filename}.")def parse_args():    parser = argparse.ArgumentParser()    parser.add_argument("--seed", type=int, default=1)    return parser.parse_args()def run_experiment(args):    # It's all Cholesky-based for this experiment    # impl = fpx.impl_covariance_based()    impl = fpx.impl_cholesky_based()    run = run_experiment_single_seed(impl)    key = jax.random.PRNGKey(args.seed)    keys = jax.random.split(key, num=10)    return run(key)    # return jax.vmap(run)(keys)def run_experiment_single_seed(impl):    def run(key):        results = {}        num_dims = 1        for num_steps in [2**i for i in range(0, 20, 2)]:            # Set up an SSM            ts = jnp.linspace(0, 1.0, num=num_steps + 1)            ssm = fpx.ssm_ode(ts, diffusion=1.0, impl=impl)            # sample = fpx.compute_stats_sample(impl=impl)            # data = sample_data(key, ssm=ssm, sample=sample)            data = jnp.sin(ts[1:])[:, None] * jnp.ones((1, num_dims))            # print(data.shape)            assert not jnp.any(jnp.isnan(data))            # Compute a reference solution            estimate = jax.jit(fpx.compute_fixedpoint_via_smoother(impl=impl))            result, _ = estimate(data, ssm)            result = impl.rv_to_mvnorm(result)            result_flat = jax.flatten_util.ravel_pytree(result)[0]            # isnan = jnp.any(jnp.isnan(result_flat))            # print(num_steps, isnan)            estimate = jax.jit(fpx.compute_fixedpoint_via_filter(impl=impl))            result, _ = estimate(data, ssm)            result = impl.rv_to_mvnorm(result)            print(result)            result_flat2 = jax.flatten_util.ravel_pytree(result)[0]            # isnan = jnp.any(jnp.isnan(result_flat))            # print(num_steps, isnan)            print(num_steps, jnp.max(jnp.abs(result_flat - result_flat2)))        return results    return rundef sample_data(key, *, ssm, sample):    (latent, data) = sample(key, ssm)    return datadef root_mean_square_error(a, b):    error_abs = jnp.abs(a - b)    return jnp.linalg.norm(error_abs / (1e-3 + jnp.abs(b))) / jnp.sqrt(b.size)if __name__ == "__main__":    main()