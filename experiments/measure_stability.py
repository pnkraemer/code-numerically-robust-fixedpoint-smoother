"""Measure the (loss of) numerical stability in fixed-point-smoothers."""import osimport pickleimport jaximport jax.numpy as jnpimport tqdmfrom fpx import eval_utils, fpxdef main(save=True) -> None:    """Estimate the loss of precision of different fixedpoint smoothers."""    # Double precision for all simulations    jax.config.update("jax_enable_x64", True)    # Choose parameters    num_dims = 1    num_steps_list = [2**i for i in range(3, 19, 2)]    # Run the experiment    results: dict = {"Cholesky-based": {}, "Covariance-based": {}}    for num_steps in tqdm.tqdm(num_steps_list):        # Translate num_steps in to data        ts = jnp.linspace(0, 1.0, num=num_steps + 1)        data = jnp.zeros((num_steps, num_dims))        # Reference: Cholesky-based Kalman filter        estimation_method = fpx.compute_fixedpoint_via_filter        impl = fpx.impl_cholesky_based()        reference = estimate_fixedpoint(ts, data, method=estimation_method, impl=impl)        # Cholesky-based fixedpoint:        estimation_method = fpx.compute_fixedpoint        impl = fpx.impl_cholesky_based()        cholbased = estimate_fixedpoint(ts, data, method=estimation_method, impl=impl)        error_cholesky = compute_error(cholbased, reference)        results["Cholesky-based"][num_steps] = error_cholesky        # Covariance-based fixedpoint:        impl = fpx.impl_covariance_based()        ts = jnp.linspace(0, 1.0, num=num_steps + 1)        covbased = estimate_fixedpoint(ts, data, method=estimation_method, impl=impl)        error_covariance = compute_error(covbased, reference)        results["Covariance-based"][num_steps] = error_covariance    if save:        filename = eval_utils.filename_results(__file__, replace="experiments")        with open(filename, "wb") as f:            pickle.dump(results, f)        print(f"\nSaved results to {filename}.\n")def estimate_fixedpoint(ts, data, *, method, impl) -> jax.Array:    """Estimate the fixedpoint."""    ssm = fpx.ssm_seventh_order_wiener_interpolation(ts, impl=impl)    estimate = jax.jit(method(impl=impl))    reference, _ = estimate(data, ssm)    reference = reference.mean    reference = jax.flatten_util.ravel_pytree(reference)[0]    return referencedef compute_error(a: jax.Array, b: jax.Array) -> jax.Array:    error_abs = jnp.abs(a - b)    return jnp.linalg.norm(error_abs) / jnp.sqrt(b.size)if __name__ == "__main__":    main()