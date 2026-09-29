#important functions
def modify_driver_list(driver_list, total_eff=False):

    controls_dict = {}

    for driver in driver_list:

        if not total_eff:
            controls = [d for d in driver_list if d != driver]

            # A-SAM has to be removed for symmetric response quantifications
            if 'A_SAM' in driver_list and driver in ['SPV', 'VB', 'S_SAM']:
                if 'A_SAM' in controls:
                    controls.remove('A_SAM')

            # Vortex and S-SAM need to be removed for asymmetric response
            elif ('SPV' in driver_list or 'VB' in driver_list) and driver == 'A_SAM':
                controls = [
                    c for c in controls
                    if c not in ['SPV', 'VB', 'S_SAM']
                ]

        else:
            if driver in ['IOD', 'IOBW']:
                controls = ['ENSO']
            elif driver == 'SPV':
                controls = ['ENSO', 'IOD']
            elif driver == 'VB':
                controls = ['ENSO', 'IOBW']
            else:
                controls = None

        controls_dict[driver] = controls

    return controls_dict

def bootstrap_regression_cell(
    y,
    drivers,
    driver_vars,
    n_boot=500,
    sample_size=200,
    add_intercept=True,
    seed=42,
    total_eff=False,
):

    n_samples = y.shape[0]
    n_drivers = len(driver_vars)

    # ------------------------------------------------------------
    # Driver dictionary
    # ------------------------------------------------------------

    driver_data = {
        var: drivers[j, :]
        for j, var in enumerate(driver_vars)
    }

    # ------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------

    controls_dict = modify_driver_list(
        driver_vars,
        total_eff=total_eff
    )

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------

    r2_out = np.full(
        (n_boot, n_drivers),
        np.nan,
        dtype=np.float64
    )

    coef_out = np.full(
        (n_boot, n_drivers),
        np.nan,
        dtype=np.float64
    )

    # ------------------------------------------------------------
    # RNG
    #
    # Important: different spatial cells get different streams.
    # ------------------------------------------------------------

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------
    # Loop over focal drivers
    # ------------------------------------------------------------

    for d, driver in enumerate(driver_vars):

        controls = controls_dict[driver]

        if controls is None:
            regression_vars = [driver]
        else:
            regression_vars = [driver] + controls

        # --------------------------------------------------------
        # Construct X
        # --------------------------------------------------------

        X = np.column_stack([
            driver_data[var]
            for var in regression_vars
        ])

        # --------------------------------------------------------
        # Remove NaNs once
        # --------------------------------------------------------

        valid = np.isfinite(y)

        valid &= np.all(
            np.isfinite(X),
            axis=1
        )

        X = X[valid]
        y_valid = y[valid]

        n_valid = len(y_valid)

        n_predictors = X.shape[1]
        n_parameters = n_predictors + int(add_intercept)

        if n_valid <= n_parameters:
            continue

        # --------------------------------------------------------
        # Bootstrap indices
        # --------------------------------------------------------

        indices = rng.integers(
            0,
            n_valid,
            size=(n_boot, sample_size)
        )

        # --------------------------------------------------------
        # Bootstrap regression
        #

        for b in range(n_boot):

            idx = indices[b]

            Xb = X[idx]
            yb = y_valid[idx]

            # Add intercept
            if add_intercept:
                Xreg = np.empty(
                    (sample_size, n_predictors + 1),
                    dtype=np.float64
                )

                Xreg[:, 0] = 1.0
                Xreg[:, 1:] = Xb

                focal_idx = 1

            else:
                Xreg = Xb
                focal_idx = 0

            # ----------------------------------------------------
            # Least-squares regression
            # ----------------------------------------------------

            try:
                beta, residuals, rank, s = np.linalg.lstsq(
                    Xreg,
                    yb,
                    rcond=None
                )
            except np.linalg.LinAlgError:
                continue


            coef_out[b, d] = beta[focal_idx]


            y_hat = Xreg @ beta

            ss_res = np.sum(
                (yb - y_hat) ** 2
            )

            y_mean = np.mean(yb)

            ss_tot = np.sum(
                (yb - y_mean) ** 2
            )

            if ss_tot > 0:
                r2_out[b, d] = (
                    1.0
                    - ss_res / ss_tot
                )

    return r2_out, coef_out


def bootstrap_map(target_xr,drivers_combi_xr, driver_vars, target_var, n_boot=5, sample_size=200, total_eff=False):

     # ------------------------------------------------------------
    # Make sample a single chunk
    # ------------------------------------------------------------

    if target_var=='z':
        target=target_xr[target_var].chunk({
            'sample':-1,
            'latitude':10,
            'longitude':20
        })

    else:
        target = target_xr[target_var].chunk({
        "sample": -1,
        "latitude": 6,
        "longitude": 6,
        })

    

    drivers_array = xr.concat(
        [drivers_combi_xr[var] for var in driver_vars],
        dim="driver"
    ).assign_coords(
        driver=driver_vars
    )

    r2, coef = xr.apply_ufunc(
    bootstrap_regression_cell,

    target,
    drivers_array,

    input_core_dims=[
        ["sample"],
        ["driver", "sample"],
    ],

    output_core_dims=[
        ["bootstrap", "driver"],
        ["bootstrap", "driver"],
    ],

    vectorize=True,

    output_dtypes=[
            np.float64,
            np.float64,
        ],

        kwargs={
            "driver_vars": driver_vars,
            "n_boot": n_boot,
            "sample_size": sample_size,
            "add_intercept": True,
            "seed": 42,
            "total_eff": total_eff,
        },

        dask="parallelized",

        dask_gufunc_kwargs={
            "output_sizes": {
                "bootstrap": n_boot,
                "driver": len(driver_vars),
            },
            "allow_rechunk": False,
        },
    )

    #try only returning the mean values of the bootstrap samples for plotting
    coef_mean = coef.mean(dim="bootstrap", skipna=True).transpose('driver', 'latitude', 'longitude')
    coef_std=coef.std(dim='bootstrap', skipna=True).transpose('driver', 'latitude', 'longitude')
    r2_mean = r2.mean(dim="bootstrap", skipna=True).transpose('driver', 'latitude', 'longitude')
        
    #stippling
    prob_positive = (coef > 0).mean(dim="bootstrap", skipna=True)
    prob_negative = (coef < 0).mean(dim="bootstrap", skipna=True)
    
    # | is the or operator 
    #         # use 95% CI (is more strict than 80%)
    significant = (prob_positive >= 0.95) | (prob_negative >= 0.95)
    significant=significant.transpose('driver', 'latitude', 'longitude')
    
    r2_mean=r2_mean.compute()
    coef_mean=coef_mean.compute()
    coef_std=coef_std.compute()
    significant=significant.compute()
    
    return r2_mean, coef_mean, coef_std, significant