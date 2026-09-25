import pandas as pd
import numpy as np
import seaborn as sns
import xarray as xr
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.linear_model import QuantileRegressor
import cartopy.crs as ccrs



PATH = '/climca/people/glattus/Hindcast_data_ready'

def df_xr_prep(df):
    da=df.to_xarray()
    da=da.rename({'index':'year'})
    return da

print('Loading data!')
#spring
ocean_SON_xr=xr.open_dataset(PATH+'/ENSO_IOD_SON.nc').drop_vars('lead_block').rename({'nino34':'ENSO', 'iod_index':'IOD'})
SAM_SON_xr=xr.open_dataset(PATH+'/SAM_ready_SON.nc')
SPV_SON_xr=xr.open_dataset(PATH+'/SPV_SON.nc').drop_vars(['lead_block', 'pressure_level']).rename({'u':'SPV'})

#summer
ocean_DJF_xr=xr.open_dataset(PATH+'/ENSO_IOB_DJF.nc').drop_vars('lead_block').rename({'nino34':'ENSO'})
SAM_DJF_xr=xr.open_dataset(PATH+'/SAM_ready_DJF.nc')
VB_DJF_xr=xr.open_dataset(PATH+'/VB_DOY_combined.nc').rename({'VB_DOY':'VB'})

#Temp & precip data
target_SON_xr=xr.open_dataset(PATH+'/Target_vars_SON_areas.nc')

target_DJF_xr=xr.open_dataset(PATH+'/Target_vars_DJF_areas.nc')

#function for excluding ssw years
def exclude_years(ds, years_list=[2002,2019]):
    years = ds.forecast_reference_time.dt.year.compute()

    keep = ~years.isin(years_list)

    return ds.isel(sample=keep)

######################################################################################################
def align_hindcast_arrays(
    array_list,
    years_list,
    names=None,
    sample_coords=None,
    reference=0,
    verbose=True
):
    """
    Check and align xarray Datasets/DataArrays containing hindcast samples.

    Samples are identified by a combination of coordinates, by default:

        number
        init_month
        season
        forecast_reference_time

    The function:
      1. checks dimensions and required coordinates
      2. checks coordinate equality
      3. checks whether all datasets contain the same sample identities
      4. detects ordering differences
      5. reorders all datasets to the reference dataset's sample order
      6. verifies the final alignment

    Parameters
    ----------
    array_list : list
        List of xarray Dataset/DataArray objects.

    names : list, optional
        Names for diagnostic output.

    sample_coords : list, optional
        Coordinates defining a unique hindcast sample.

    reference : int
        Index of the dataset whose sample ordering is used
        as the reference.

    verbose : bool
        Print diagnostics.

    Returns
    -------
    aligned_arrays : list
        Aligned xarray objects.
    """

    # ---------------------------------------------------------
    # 1. Defaults
    # ---------------------------------------------------------

    if sample_coords is None:
        sample_coords = [
            "number",
            "init_month",
            "season",
            "forecast_reference_time"
        ]

    n_arrays = len(array_list)

    if names is None:
        names = [
            f"array_{i}"
            for i in range(n_arrays)
        ]

    if len(names) != n_arrays:
        raise ValueError(
            "names must have the same length as array_list"
        )

    if reference < 0 or reference >= n_arrays:
        raise ValueError(
            "reference must be a valid array index"
        )

    # ---------------------------------------------------------
    # 2. Basic checks
    # ---------------------------------------------------------

    if verbose:
        print("=" * 70)
        print("BASIC STRUCTURE")
        print("=" * 70)

    for name, arr in zip(names, array_list):

        if not isinstance(
            arr,
            (xr.Dataset, xr.DataArray)
        ):
            raise TypeError(
                f"{name} is not an xarray Dataset/DataArray"
            )

        if "sample" not in arr.dims:
            raise ValueError(
                f"{name} does not contain a 'sample' dimension"
            )

        print(
            f"{name:15s} "
            f"sample size = {arr.sizes['sample']}"
        )

    # ---------------------------------------------------------
    # 3. Check sample sizes
    # ---------------------------------------------------------

    sample_sizes = [
        arr.sizes["sample"]
        for arr in array_list
    ]

    if len(set(sample_sizes)) != 1:

        raise ValueError(
            "Datasets have different sample sizes:\n"
            + "\n".join(
                f"{name}: {size}"
                for name, size in zip(
                    names,
                    sample_sizes
                )
            )
        )

    n_samples = sample_sizes[0]

    # ---------------------------------------------------------
    # 4. Check sample coordinates
    # ---------------------------------------------------------

    if verbose:
        print("\n" + "=" * 70)
        print("CHECKING SAMPLE COORDINATES")
        print("=" * 70)

    for name, arr in zip(names, array_list):

        for coord in sample_coords:

            if coord not in arr.coords:

                raise ValueError(
                    f"{name} is missing coordinate "
                    f"{coord!r}"
                )

            if arr[coord].dims != ("sample",):

                raise ValueError(
                    f"{name}[{coord!r}] has dimensions "
                    f"{arr[coord].dims}, expected ('sample',)"
                )

        print(
            f"{name:15s}: all coordinates present"
        )

    # ---------------------------------------------------------
    # 5. Construct sample identities
    # ---------------------------------------------------------

    if verbose:
        print("\n" + "=" * 70)
        print("CONSTRUCTING SAMPLE IDENTITIES")
        print("=" * 70)

    identities = []

    for name, arr in zip(names, array_list):

        identity = pd.MultiIndex.from_arrays(
            [
                arr[coord].values
                for coord in sample_coords
            ],
            names=sample_coords
        )

        identities.append(identity)

        n_unique = identity.nunique()

        print(
            f"{name:15s}: "
            f"{n_unique} unique / "
            f"{n_samples} total"
        )

        if identity.has_duplicates:

            raise ValueError(
                f"{name} contains duplicate "
                f"sample identities."
            )

    # ---------------------------------------------------------
    # 6. Compare coordinate arrays position-by-position
    # ---------------------------------------------------------

    if verbose:
        print("\n" + "=" * 70)
        print("POSITIONAL COMPARISON")
        print("=" * 70)

    ref_arr = array_list[reference]
    ref_name = names[reference]

    for coord in sample_coords:

        print(f"\n{coord}")

        for i, arr in enumerate(array_list):

            if i == reference:
                continue

            same = np.array_equal(
                ref_arr[coord].values,
                arr[coord].values
            )

            print(
                f"  {ref_name} == {names[i]}: {same}"
            )

    # ---------------------------------------------------------
    # 7. Compare sample sets
    # ---------------------------------------------------------

    if verbose:
        print("\n" + "=" * 70)
        print("SAMPLE SET COMPARISON")
        print("=" * 70)

    reference_identity = identities[reference]

    reference_set = set(reference_identity)

    for i, identity in enumerate(identities):

        if i == reference:
            continue

        current_set = set(identity)

        only_reference = (
            reference_set - current_set
        )

        only_current = (
            current_set - reference_set
        )

        same_set = (
            len(only_reference) == 0
            and len(only_current) == 0
        )

        print(
            f"\n{ref_name} vs {names[i]}"
        )

        print(
            f"Same sample set: {same_set}"
        )

        print(
            f"Only {ref_name}: "
            f"{len(only_reference)}"
        )

        print(
            f"Only {names[i]}: "
            f"{len(only_current)}"
        )

        if not same_set:

            print("\nExample samples only in reference:")

            for item in list(only_reference)[:5]:
                print(item)

            print(
                "\nExample samples only in current:"
            )

            for item in list(only_current)[:5]:
                print(item)

            raise ValueError(
                f"{names[i]} does not contain "
                f"the same hindcast samples as "
                f"{ref_name}."
            )

    print(
        "\nAll datasets contain exactly the "
        "same sample identities."
    )

    # ---------------------------------------------------------
    # 8. Reorder datasets to reference ordering
    # ---------------------------------------------------------

    if verbose:
        print("\n" + "=" * 70)
        print("REORDERING TO REFERENCE SAMPLE ORDER")
        print("=" * 70)

    aligned_arrays = []

    for i, (name, arr, identity) in enumerate(
        zip(names, array_list, identities)
    ):

        # -----------------------------------------------------
        # Find the position of every reference sample
        # in this dataset.
        # -----------------------------------------------------

        position_lookup = {
            sample_identity: position
            for position, sample_identity
            in enumerate(identity)
        }

        try:

            reorder_index = np.array(
                [
                    position_lookup[sample_identity]
                    for sample_identity
                    in reference_identity
                ],
                dtype=int
            )

        except KeyError as e:

            raise ValueError(
                f"Could not find sample {e} "
                f"in {name}"
            )

        # Reorder along original sample dimension
        arr_aligned = arr.isel(
            sample=reorder_index
        )

        aligned_arrays.append(
            arr_aligned
        )

        # Check that ordering now matches
        new_identity = pd.MultiIndex.from_arrays(
            [
                arr_aligned[coord].values
                for coord in sample_coords
            ],
            names=sample_coords
        )

        if not new_identity.equals(
            reference_identity
        ):

            raise RuntimeError(
                f"Reordering failed for {name}"
            )

        print(
            f"{name:15s}: reordered successfully"
        )

    # ---------------------------------------------------------
    # 9. Final verification
    # ---------------------------------------------------------

    if verbose:
        print("\n" + "=" * 70)
        print("FINAL VERIFICATION")
        print("=" * 70)

    for coord in sample_coords:

        reference_values = (
            aligned_arrays[reference][coord].values
        )

        for i, arr in enumerate(aligned_arrays):

            same = np.array_equal(
                reference_values,
                arr[coord].values
            )

            if not same:

                raise RuntimeError(
                    f"Final alignment failed for "
                    f"{names[i]} coordinate {coord}"
                )

    print(
        "All sample-identifying coordinates "
        "are now identical."
    )

    print(
        f"All arrays contain {n_samples} "
        "aligned samples."
    )

    print("\nAlignment successful.")

    #exclude ssw years
    aligned_arrays=[exclude_years(arr_aligned, years_list=years_list) for arr_aligned in aligned_arrays]
    print(f"All arrays contain {aligned_arrays[0].sizes['sample']} aligned samples after excluding SSW years.")

    return aligned_arrays
####################################################################################################

print('Aligning hindcast arrays!')
#spring
all_aligned=align_hindcast_arrays([target_SON_xr, 
                                   ocean_SON_xr, SPV_SON_xr, SAM_SON_xr], 
                                  years_list=[2002,2019],
                                  names=['Target',
                                          'ocean', 'SPV', 'SAM'])

target_SON_xr=all_aligned[0]
drivers_aligned=all_aligned[1:]
drivers_SON_xr=xr.merge(drivers_aligned)
print(target_SON_xr.sizes)
print('Spring aligned!')

#summer
#slightly different approach here due to init month restriction
init_sel = np.intersect1d(
    np.unique(VB_DJF_xr.init_month.values),
    np.unique(target_DJF_xr.init_month.values)
)
print('Init months in all necessary Arrays')
print(init_sel)

djf_list = [
    target_DJF_xr.drop_vars(['lead_block', 'season']),
    ocean_DJF_xr.drop_vars(['season']),
    VB_DJF_xr,
    SAM_DJF_xr.drop_vars([ 'season'])
]

djf_sel_list = [
    array.where(
        array.init_month.isin(init_sel),
        drop=True
    )
    for array in djf_list
]

def get_common_identity(arr):
    return pd.MultiIndex.from_arrays(
        [
            arr.number.values,
            arr.init_month.values,
            arr.forecast_reference_time.values
        ],
        names=[
            'number',
            'init_month',
            'forecast_reference_time'
        ]
    )

identities = [
    get_common_identity(arr)
    for arr in djf_sel_list
]

common_ids = identities[0]

for ids in identities[1:]:
    common_ids = common_ids.intersection(ids)

print("Number of common samples:", len(common_ids))

def add_sample_id(arr):
    return arr.assign_coords(
        sample_id=(
            "sample",
            [
                f"{n}_{t}"
                for n, t in zip(
                    arr.number.values,
                    arr.forecast_reference_time.values
                )
            ]
        )
    )

djf_sel_list = [
    add_sample_id(arr)
    for arr in djf_sel_list
]

common_ids = set(djf_sel_list[0].sample_id.values)

for arr in djf_sel_list[1:]:
    common_ids &= set(arr.sample_id.values)

print(len(common_ids))

common_ids = list(common_ids)

aligned = [
    exclude_years(arr.where(
        arr.sample_id.isin(common_ids),
        drop=True
    ), [2002,2003,2019,2020])
    for arr in djf_sel_list
]


#all_aligned_DJF=align_hindcast_arrays([target_DJF_xr, z500_DJF_xr, ocean_DJF_xr, VB_DJF_xr, SAM_DJF_xr], 
#                                  years_list=[2003,2020],
#                                  names=['Target','z500', 'ocean', 'SPV', 'SAM'])

target_DJF_xr=aligned[0]
drivers_aligned_DJF=aligned[1:]
drivers_DJF_xr=xr.merge(drivers_aligned_DJF)

print(target_DJF_xr.sizes)

print('Regression maps start')

import sys
sys.path.insert(0, "../whole_period_effects")  # add Folder_2 path to search list

from Functions import plot_map, subplots_map, plot_map_circ, subplots_map_circ

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
    r2_mean = r2.mean(dim="bootstrap", skipna=True).transpose('driver', 'latitude', 'longitude')
    
    #stippling
    prob_positive = (coef > 0).mean(dim="bootstrap", skipna=True)
    prob_negative = (coef < 0).mean(dim="bootstrap", skipna=True)

    # | is the or operator 
    significant = (prob_positive >= 0.8) | (prob_negative >= 0.8)

    significant=significant.transpose('driver', 'latitude', 'longitude')

    r2_mean=r2_mean.compute()
    coef_mean=coef_mean.compute()
    significant=significant.compute()

    return r2_mean, coef_mean, significant

#spring regressions
driver_vars_SON = ["ENSO", "IOD", 'SPV', 'A_SAM', 'S_SAM']
driver_vars_tot=['ENSO', 'IOD', 'SPV']

#Precip SON direct effect
r2_SON_precip_direct, coef_SON_precip_direct, significant_SON_precip_direct=bootstrap_map(target_SON_xr, drivers_SON_xr, 
                                                           driver_vars_SON, target_var='tp', n_boot=100, 
                                                        sample_size=200, total_eff=False)
#Precip SON total effect
r2_SON_precip_total, coef_SON_precip_total, significant_SON_precip_total=bootstrap_map(target_SON_xr, drivers_SON_xr,
                                                           driver_vars_tot, target_var='tp', n_boot=100,
                                                        sample_size=200, total_eff=True)
#Coef maps
subplots_map(ds=coef_SON_precip_direct, title_list=driver_vars_SON, cmap=plt.cm.BrBG, unit='mm', steps=0.1, \
                 cbar_each=None, heading='Mean regression coefficients Precipitation SON',
                   stations=None, BF=significant_SON_precip_direct)
plt.close()

subplots_map(ds=coef_SON_precip_total, title_list=driver_vars_tot, cmap=plt.cm.BrBG, unit='mm', steps=0.1, \
                 cbar_each=None, heading='Mean regression coefficients Precipitation Total SON',
                   stations=None, BF=significant_SON_precip_total)
plt.close()

#R2 precip SON maps
cmap_r2=plt.cm.PuOr
subplots_map(ds=r2_SON_precip_direct, title_list=driver_vars_SON, cmap=cmap_r2, unit=' ', steps=0.1, \
                 cbar_each=None, heading='Mean R2 Precipitation SON',
                   stations=None, R2_plot=True)
plt.close()


subplots_map(ds=r2_SON_precip_total, title_list=driver_vars_tot, cmap=cmap_r2, unit=' ', steps=0.1, \
                 cbar_each=None, heading='Mean R2 Precipitation Total SON',
                   stations=None, R2_plot=True)
plt.close()

print('Precip SON maps done!')

r2_SON_temp_direct, coef_SON_temp_direct, significant_SON_temp_direct=bootstrap_map(target_SON_xr, drivers_SON_xr, 
                                                       driver_vars_SON, target_var='t2m', n_boot=100,
                                                         sample_size=200, total_eff=False)

r2_SON_temp_total, coef_SON_temp_total, significant_SON_temp_total=bootstrap_map(target_SON_xr, drivers_SON_xr,
                                                     driver_vars_tot, target_var='t2m', n_boot=100,
                                                        sample_size=200, total_eff=True)

subplots_map(ds=coef_SON_temp_direct, title_list=driver_vars_SON, cmap=plt.cm.RdBu_r, unit='K', steps=0.1, \
                 cbar_each=None, heading='Mean regression coefficients Temperature SON',
                   stations=None, BF=significant_SON_temp_direct)
plt.close()

subplots_map(ds=coef_SON_temp_total, title_list=driver_vars_tot, cmap=plt.cm.RdBu_r, unit='K', steps=0.1, \
                 cbar_each=None, heading='Mean regression coefficients Temperature Total SON',
                   stations=None, BF=significant_SON_temp_total)
plt.close()

#R2 Temp maps

subplots_map(ds=r2_SON_temp_direct, title_list=driver_vars_SON, cmap=cmap_r2, unit=' ', steps=0.1, \
                 cbar_each=None, heading='Mean R2 Temperature SON',
                   stations=None, R2_plot=True)
plt.close()

subplots_map(ds=r2_SON_temp_total, title_list=driver_vars_tot, cmap=cmap_r2, unit=' ', steps=0.1, \
                 cbar_each=None, heading='Mean R2 Temperature Total SON',
                   stations=None, R2_plot=True)
plt.close()


print('Temp SON maps done!')


print('Start DJF maps!')

driver_vars_DJF = ["ENSO", "IOBW", 'VB', 'A_SAM', 'S_SAM']
driver_vars_tot_DJF=['ENSO', 'IOBW', 'VB']

#Precip DJF direct effect
r2_DJF_precip_direct, coef_DJF_precip_direct, significant_DJF_precip_direct=bootstrap_map(target_DJF_xr, drivers_DJF_xr, 
                                                           driver_vars_DJF, target_var='tp', n_boot=100, 
                                                        sample_size=200, total_eff=False)
#Precip DJF total effect
r2_DJF_precip_total, coef_DJF_precip_total, significant_DJF_precip_total=bootstrap_map(target_DJF_xr, drivers_DJF_xr,
                                                           driver_vars_tot_DJF, target_var='tp', n_boot=100,
                                                        sample_size=200, total_eff=True)

subplots_map(ds=coef_DJF_precip_direct, title_list=driver_vars_DJF, cmap=plt.cm.BrBG, unit='mm', steps=0.1, \
                 cbar_each=None, heading='Mean regression coefficients Precipitation DJF',
                   stations=None, BF=significant_DJF_precip_direct)
plt.close()

subplots_map(ds=coef_DJF_precip_total, title_list=driver_vars_tot_DJF, cmap=plt.cm.BrBG, unit='mm', steps=0.1, \
                 cbar_each=None, heading='Mean regression coefficients Precipitation Total DJF',
                   stations=None, BF=significant_DJF_precip_total)
plt.close()

subplots_map(ds=r2_DJF_precip_direct, title_list=driver_vars_DJF, cmap=cmap_r2, unit=' ', steps=0.1, \
                 cbar_each=None, heading='Mean R2 Precipitation DJF',
                   stations=None, R2_plot=True)
plt.close()

subplots_map(ds=r2_DJF_precip_total, title_list=driver_vars_tot_DJF, cmap=cmap_r2, unit=' ', steps=0.1, \
                 cbar_each=None, heading='Mean R2 Precipitation Total DJF',
                   stations=None, R2_plot=True)
plt.close()

print('Precip DJF maps done!')

r2_DJF_temp_direct, coef_DJF_temp_direct, significant_DJF_temp_direct=bootstrap_map(target_DJF_xr, drivers_DJF_xr, 
                                                       driver_vars_DJF, target_var='t2m', n_boot=100,
                                                         sample_size=200, total_eff=False)

r2_DJF_temp_total, coef_DJF_temp_total, significant_DJF_temp_total=bootstrap_map(target_DJF_xr, drivers_DJF_xr,
                                                     driver_vars_tot_DJF, target_var='t2m', n_boot=100,
                                                        sample_size=200, total_eff=True)

subplots_map(ds=coef_DJF_temp_direct, title_list=driver_vars_DJF, cmap=plt.cm.RdBu_r, unit='K', steps=0.1, \
                 cbar_each=None, heading='Mean regression coefficients Temperature DJF',
                   stations=None, BF=significant_DJF_temp_direct)
plt.close()

subplots_map(ds=coef_DJF_temp_total, title_list=driver_vars_tot_DJF, cmap=plt.cm.RdBu_r, unit='K', steps=0.1, \
                 cbar_each=None, heading='Mean regression coefficients Temperature Total DJF',
                   stations=None, BF=significant_DJF_temp_total)
plt.close()

subplots_map(ds=r2_DJF_temp_direct, title_list=driver_vars_DJF, cmap=cmap_r2, unit=' ', steps=0.1, \
                 cbar_each=None, heading='Mean R2 Temperature DJF',
                   stations=None, R2_plot=True)
plt.close()

subplots_map(ds=r2_DJF_temp_total, title_list=driver_vars_tot_DJF, cmap=cmap_r2, unit=' ', steps=0.1, \
                 cbar_each=None, heading='Mean R2 Temperature Total DJF',
                   stations=None, R2_plot=True)
plt.close()

print('Temp DJF maps done!')


print('All regression maps done!')