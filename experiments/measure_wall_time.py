"""Measure the wall time of fixedpoint estimation via differnet methods."""import jax.numpy as jnpimport jaximport timeimport pickleimport osfrom fpx import fpximport argparsedef main():    args = parse_args()    results = run_experiment(args)    dirname = str(__file__)    dirname = dirname.replace("experiments", "results")    dirname = dirname.replace(".py", "")    os.makedirs(dirname, exist_ok=True)    filename = f"{dirname}/results.pkl"    with open(filename, "wb") as f:        pickle.dump(results, f)    print(f"Saved results to {filename}.")def parse_args():    parser = argparse.ArgumentParser()    parser.add_argument("--num_runs", type=int, default=2)    parser.add_argument("--seed", type=int, default=1)    return parser.parse_args()# todo: make the estimators accept parametrised ssms?#  this way, we can truly compute things on the fly and the smoother looks a bit worse?# todo: can we choose a reasonably high-dimensional SSM?#  Or should we compute targets at multiple states? The filter still looks too good...def run_experiment(args):    key = jax.random.PRNGKey(args.seed)    # It's all Cholesky-based for this experiment    impl = fpx.impl_cholesky_based()    results = {}    for num_steps in [1_000]:        label_num_steps = f"$K={num_steps}$"        results[label_num_steps] = {}        print(label_num_steps)        print("-" * len(label_num_steps))        for num_dims in [2]:            label_num_dims = f"$d={num_dims}$"            results[label_num_steps][label_num_dims] = {}            print()            print(label_num_dims)            print("-" * len(label_num_dims))            # Set up an SSM            ts = jnp.linspace(0, 1, num=num_steps)            ssm = fpx.ssm_car_tracking_velocity(                ts, noise=1e-4, diffusion=1.0, impl=impl, dim=num_dims            )            sample = fpx.compute_stats_sample(impl=impl)            data = sample_data(key, ssm=ssm, sample=sample)            assert not jnp.any(jnp.isnan(data))            # Compute a reference solution            estimate = jax.jit(fpx.compute_fixedpoint(impl=impl))            ref, _ = estimate(data, ssm)            def callback(x):                return {"size": jax.flatten_util.ravel_pytree(x)[0].size}            via_filter = fpx.compute_fixedpoint_via_filter(impl=impl, callback=callback)            via_fixedinterval = fpx.compute_fixedpoint_via_smoother(                impl=impl, callback=callback            )            via_recursion = fpx.compute_fixedpoint(impl=impl, callback=callback)            for name, estimate in [                ("Via recursion", via_recursion),                ("Via fixed-interval", via_fixedinterval),                ("Via filter", via_filter),            ]:                print(name)                estimate = jax.jit(estimate)                t = benchmark(                    estimate, ref=ref, ssm=ssm, data=data, num_runs=args.num_runs                )                print(f"\t {min(t):.1e}")                results[label_num_steps][label_num_dims][name] = min(t)        print()    return resultsdef sample_data(key, *, ssm, sample):    (latent, data) = sample(key, ssm)    return datadef benchmark(fixedpoint, *, ref, data, ssm, num_runs):    # Execute once to pre-compile (and to compute errors)    initial_rts, _aux = fixedpoint(data, ssm)    if jnp.any(jnp.isnan(initial_rts.mean)):        print("NaN detected")        return [-1.0]    # If the values don't match, abort    rmse = root_mean_square_error(initial_rts.mean, ref.mean)    if rmse > 1e-2:        print(f"Values do not match the reference: rmse={rmse:.1e}")        return [-1.0]    ts = []    for _ in range(num_runs):        t0 = time.perf_counter()        initial_rts, _ = fixedpoint(data, ssm)        initial_rts.mean.block_until_ready()        t1 = time.perf_counter()        ts.append(t1 - t0)    return tsdef root_mean_square_error(a, b):    error_abs = jnp.abs(a - b)    return jnp.linalg.norm(error_abs / (1e-3 + jnp.abs(b))) / jnp.sqrt(b.size)if __name__ == "__main__":    main()