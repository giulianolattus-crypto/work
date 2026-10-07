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


PATH = '/climca/people/glattus/Hindcast_data_ready'

def df_xr_prep(df):
    da=df.to_xarray()
    da=da.rename({'index':'year'})
    return da

##La Plata
#SON
hind_SON_LaPlata_xr=xr.open_dataset(PATH+'/Target_vars_LP_SON.nc')

#DJF
hind_DJF_LaPlata_xr=xr.open_dataset(PATH+'/Target_vars_LP_DJF.nc')

##Andes
hind_SON_Andes_xr=xr.open_dataset(PATH+'/Target_vars_Andes_SON.nc')
hind_DJF_Andes_xr=xr.open_dataset(PATH+'/Target_vars_Andes_DJF.nc')

#spring
ocean_SON_xr=xr.open_dataset(PATH+'/ENSO_IOD_SON.nc').drop_vars('lead_block').rename({'nino34':'ENSO', 'iod_index':'IOD'})
SAM_SON_xr=xr.open_dataset(PATH+'/SAM_ready_SON.nc')
SPV_SON_xr=xr.open_dataset(PATH+'/SPV_SON.nc').drop_vars(['lead_block', 'pressure_level']).rename({'u':'SPV'})

#summer
ocean_DJF_xr=xr.open_dataset(PATH+'/ENSO_IOB_DJF.nc').drop_vars('lead_block').rename({'nino34':'ENSO'})
SAM_DJF_xr=xr.open_dataset(PATH+'/SAM_ready_DJF.nc')
VB_DJF_xr=xr.open_dataset(PATH+'/VB_DOY_combined.nc').rename({'VB_DOY':'VB'})


#Functions for aligning arrays
#function for excluding ssw years
def exclude_years(ds, years_list=[2002,2019]):
    years = ds.forecast_reference_time.dt.year.compute()

    keep = ~years.isin(years_list)

    return ds.isel(sample=keep)

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

#aligning SON
all_aligned=align_hindcast_arrays([hind_SON_LaPlata_xr, hind_SON_Andes_xr,ocean_SON_xr, SPV_SON_xr, SAM_SON_xr], 
                                  names=['LaPlata','Andes','ocean', 'SPV', 'SAM'], years_list=[2002,2019])
target_Andes_SON_xr=all_aligned[1]
target_LP_SON_xr=all_aligned[0]
drivers_aligned=all_aligned[2:]
drivers_SON_xr=xr.merge(drivers_aligned)

#aligning DJF

init_sel = np.intersect1d(
    np.unique(VB_DJF_xr.init_month.values),
    np.unique(hind_DJF_LaPlata_xr.init_month.values)
)
print(init_sel)

djf_list = [
    hind_DJF_LaPlata_xr.drop_vars(['lead_block', 'season']),
    hind_DJF_Andes_xr.drop_vars(['lead_block', 'season']),
    ocean_DJF_xr.drop_vars(['season']),
    VB_DJF_xr,
    SAM_DJF_xr.drop_vars([ 'season'])
]

djf_sel_list = [
    exclude_years(array.where(
        array.init_month.isin(init_sel),
        drop=True
    ), [2002,2003,2019,2020])
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
    arr.where(
        arr.sample_id.isin(common_ids),
        drop=True
    )
    for arr in djf_sel_list
]

#alignment without VB index
aligned_DJF_no_VB=align_hindcast_arrays([hind_DJF_LaPlata_xr, hind_DJF_Andes_xr, ocean_DJF_xr, SAM_DJF_xr], years_list=[2003,2020])
target_Andes_DJF_xr_no_VB=aligned_DJF_no_VB[1]
target_LP_DJF_xr_no_VB=aligned_DJF_no_VB[0]
drivers_aligned_no_VB=aligned_DJF_no_VB[2:]
drivers_DJF_xr_no_VB=xr.merge(drivers_aligned_no_VB)

#alignment with VB index
target_Andes_DJF_xr=aligned[1]
target_LP_DJF_xr=aligned[0]
drivers_aligned_DJF=aligned[2:]
drivers_DJF_xr=xr.merge(drivers_aligned_DJF)

######################################################################################################
#Regression analysis
#######################################################################################################
print('Start regressions')

import sys
sys.path.insert(0, "../whole_period_effects")  # add Folder_2 path to search list

import Functions

from drawing_dag import plot_causal_networks_grid

drivers_SON= ['ENSO', 'IOD', 'SPV', 'S_SAM', 'A_SAM']
drivers_tot_SON=['ENSO', 'IOD', 'SPV']

def bootstrap_regression(
    target_ds,
    driver_ds,
    target_var,
    driver_vars,
    n_boot=50,
    sample_size=10000,
    add_intercept=True,
    printer=False,
    seed=42,
    quantiles=None,
    total_eff=False
):
    """
    target_var: string, e.g. "tprate" or "t2m"
    driver_vars: list of strings, e.g. ["nino34"] or ["nino34", "iod_index"]
    """

    results = []

    n_samples = target_ds.sizes["sample"]

    # Check that target and drivers have same number of samples
    for var in driver_vars:
        if driver_ds.sizes["sample"] != n_samples:
            raise ValueError(
                f"Target and driver sample sizes differ: "
                f"target={n_samples}, {var}={driver_ds.sizes['sample']}"
            )

    #include driver_vars selection as in bachelor thesis
    controls_dict = modify_driver_list(
        driver_vars,
        total_eff=total_eff
    )

    #rerun yields the same result
    rng = np.random.default_rng(seed)

    #get bootstrap indices before opening loop to ensure each driver regression has same samples
    bootstrap_indices = [
        rng.choice(
            n_samples,
            size=sample_size,
            replace=True
        )
        for _ in range(n_boot)
    ]

        
    #initalize counter on how many bootstraps have enough values
    counter=0
    for driver in driver_vars:
        controls = controls_dict[driver]

        #special case handling: ENSO total effect needs no controls
        if controls is None:
            regression_vars=[driver]
        else:
            regression_vars = [driver] + controls
        if ~printer:
            print(driver, regression_vars)
            
        for i, idx in enumerate(bootstrap_indices):
    

            y = target_ds[target_var].isel(sample=idx).values

            X = pd.DataFrame({
            var: driver_ds[var].isel(sample=idx).values
            for var in regression_vars
            })

            df = X.copy()
            df["target"] = y
            df = df.dropna()

            # Number of valid observations
            n_valid = len(df)

            # Not enough data to fit regression
            if n_valid == 0:
                print(f"Bootstrap {i}: 0 valid samples -- skipping")
                continue

            # Need at least more observations than parameters
            n_predictors = len(regression_vars)
            n_parameters = n_predictors + int(add_intercept)

            if n_valid <= n_parameters:
                if printer:
                    print(
                    f"Bootstrap {i}: only {n_valid} valid samples "
                    f"for {n_parameters} parameters -- skipping"
                    )
                else:
                    counter+=1
                continue

            X_clean = df[regression_vars]
            y_clean = df["target"]

            if add_intercept:
                X_clean = sm.add_constant(X_clean)

        
            
            if quantiles is None:
                model = sm.OLS(y_clean, X_clean).fit()

                row = {
                                    "bootstrap": i,
                                    "target": target_var,
                                    "drivers": "+".join(regression_vars),
                                    "r2": model.rsquared
                                }
                for name, value in model.params.items():
                            if name==driver:
                                row[f"coef_{name}"] = value

            else:
                #quantiles = [0.05, 0.5, 0.95]
                
                for quantile in quantiles:
                    qr = QuantileRegressor(quantile=quantile, fit_intercept=add_intercept, alpha=0)
                    model = qr.fit(X_clean, y_clean)

                    row = {
                                        "bootstrap": i,
                                        "target": target_var,
                                        "drivers": "+".join(regression_vars),
                                        f"r2_{quantile}": model.score
                                    }
                    
                    # Pairs column names directly with their respective slope
                    for col_name, coef_val in zip(X.columns, model.coef_):
                        row[f'coef_{col_name}_{quantile}'] = coef_val
                        
                    if add_intercept:
                        row[f'intercept_{quantile}'] = model.intercept_

    
        

            results.append(row)

    if ~printer:
        print('Valid bootstraps: '+ str(n_boot-counter)+'/'+str(n_boot))
    return pd.DataFrame(results)

from Functions_bootstrap import modify_driver_list

def plot_bootstrap_coefficients(results, driver_vars, title="Bootstrap regression coefficients", 
                                bins_num=100, era5_vals=None):
    'era5_vals must be a dict with driver names as in driver_vars and values from ERA5 regression'

    fsize=18
    lim_x=5.0
    ncols=len(driver_vars)

    global_max=max(
        np.max(np.abs(results[f"coef_{driver}"]))
        for driver in driver_vars
    )

    fig, axes = plt.subplots(
    ncols=ncols,
    figsize=(ncols * 6, 6),
    sharey=True,
    squeeze=False
    )

    axes = axes.ravel()

    for i, var in enumerate(driver_vars):
        n, bins, _=axes[i].hist(
            results[f"coef_{var}"],
            bins=bins_num #,
            #label=var
        )

        subplot_title=var

        mean = results[f"coef_{var}"].mean()
        std = results[f"coef_{var}"].std()

        #add CI
        lower = np.nanpercentile(results[f'coef_{var}'], 2.5)
        upper = np.nanpercentile(results[f'coef_{var}'], 97.5)

        significant = (lower > 0) or (upper < 0)

        axes[i].axvline(
                    mean,
                    linestyle="--", color='red', 
                    label=f'Mean: {mean:.2f}'
                )

        if era5_vals is not None:
            axes[i].axvline(era5_vals[var], color='black', label=f'ERA5: {era5_vals[var]:.2f}', linewidth=3)

        #axes[i].axvspan(xmin=mean-std, xmax=mean+std, alpha=.3, color='red', label='$\pm 1 \sigma$: '+str(round(std,2)))
        axes[i].axvspan(xmin=lower, xmax=upper, alpha=.3, color='red', label=f'95% CI: [{lower:.2f}, {upper:.2f}]')

        if significant:
            #or maybe add text to driver heading
            subplot_title+=' (significant)'
            # axes[i].text(
            #     0.8, 0.15,
            #     "Significant \n link",
            #     transform=axes[i].transAxes,
            #     ha="center",
            #     va="top",
            #     fontsize=18
            # )

        axes[i].set_xlabel(f"Reg coef of {var}", fontsize=fsize)
        axes[i].tick_params( axis="both",
                    which="major",
                    labelsize=fsize)
        axes[i].grid(alpha=.3)
        axes[i].legend(fontsize=fsize)
        #restrict x axis to some value
        if global_max>lim_x:
            subplot_title+=' (*)'

        else:
            lim_x=global_max
    
        axes[i].set_xlim(-lim_x,lim_x)
        axes[i].set_title(subplot_title, fontsize=fsize, fontweight='bold')

    axes[0].set_ylabel("Count", fontsize=fsize)
    fig.suptitle(title, fontsize=fsize+4, fontweight='heavy', y=1.1)
    fig.tight_layout()
    fig.savefig(title.replace(" ", "_")+".jpg", dpi=300, bbox_inches='tight')
    plt.show()

#####################################################################################
#1. Mixing all init months and years
####################################################################################

#Spring (SON) direct effects on T and precip in both regions
era5_temp_Andes_direct={'ENSO':0.11, 'IOD':-0.08, 'SPV':0.06, 'S_SAM':0.01, 'A_SAM':0.36}
era5_precip_Andes_direct={'ENSO':0.14, 'IOD':-0.02, 'SPV':-0.20, 'S_SAM':0.12, 'A_SAM':-0.17}
era5_temp_LP_direct={'ENSO':-0.34, 'IOD':0.60, 'SPV':-0.06, 'S_SAM':0.02, 'A_SAM':-0.34}
era5_precip_LP_direct={'ENSO':0.53, 'IOD':-0.15, 'SPV':-0.10, 'S_SAM':-0.15, 'A_SAM':-0.21}

era5_val_direct_list=[era5_temp_Andes_direct, era5_precip_Andes_direct, era5_temp_LP_direct, era5_precip_LP_direct]
heading_list=['Andes T','Andes Pr','La Plata T','La Plata Pr']
#direct effects on T and precip in both regions
results_direct_T_Andes_SON_1000=bootstrap_regression(target_Andes_SON_xr, drivers_SON_xr, 't2m', drivers_SON, n_boot=10000, sample_size=200, add_intercept=False)
results_direct_Pr_Andes_SON_1000=bootstrap_regression(target_Andes_SON_xr, drivers_SON_xr, 'tp', drivers_SON, n_boot=10000, sample_size=200, add_intercept=False)
results_direct_T_LP_SON_1000=bootstrap_regression(target_LP_SON_xr, drivers_SON_xr, 't2m', drivers_SON, n_boot=10000, sample_size=200, add_intercept=False)
results_direct_Pr_LP_SON_1000=bootstrap_regression(target_LP_SON_xr, drivers_SON_xr, 'tp', drivers_SON, n_boot=10000, sample_size=200, add_intercept=False)

for i, element in enumerate([results_direct_T_Andes_SON_1000, results_direct_Pr_Andes_SON_1000, results_direct_T_LP_SON_1000, results_direct_Pr_LP_SON_1000]):
    plot_bootstrap_coefficients(element, drivers_SON,
                                title=f'Bootstrap regression coefficients \n {heading_list[i]}', 
                                era5_vals=era5_val_direct_list[i])
    
#total effects on T and precip in both regions
era5_temp_Andes_total={'ENSO':-0.1, 'IOD':-0.18, 'SPV':0.05}
era5_precip_Andes_total={'ENSO':0.21, 'IOD':0.00, 'SPV':-0.11}
era5_temp_LP_total={'ENSO':-0.14, 'IOD':0.69, 'SPV':-0.04}
era5_precip_LP_total={'ENSO':0.65, 'IOD':-0.08, 'SPV':0.00}

era5_val_total_list=[era5_temp_Andes_total, era5_precip_Andes_total, 
                     era5_temp_LP_total, era5_precip_LP_total]


tot_drivers=['ENSO', 'IOD', 'SPV']
results_tot_T_Andes_SON_1000=bootstrap_regression(target_Andes_SON_xr, drivers_SON_xr, 't2m', tot_drivers, n_boot=10000, sample_size=200, add_intercept=False, total_eff=True)
results_tot_Pr_Andes_SON_1000=bootstrap_regression(target_Andes_SON_xr, drivers_SON_xr, 'tp', tot_drivers, n_boot=10000, sample_size=200, add_intercept=False, total_eff=True)
results_tot_T_LP_SON_1000=bootstrap_regression(target_LP_SON_xr, drivers_SON_xr, 't2m', tot_drivers, n_boot=10000, sample_size=200, add_intercept=False, total_eff=True)
results_tot_Pr_LP_SON_1000=bootstrap_regression(target_LP_SON_xr, drivers_SON_xr, 'tp', tot_drivers, n_boot=10000, sample_size=200, add_intercept=False, total_eff=True)

for i, element in enumerate([results_tot_T_Andes_SON_1000, results_tot_Pr_Andes_SON_1000, results_tot_T_LP_SON_1000, results_tot_Pr_LP_SON_1000]):
    plot_bootstrap_coefficients(element, tot_drivers, 
                                title=f'Bootstrap regression coefficients \n Total effects '+ f' {heading_list[i]}',
                                era5_vals=era5_val_total_list[i])

#other SON links
results_IOD_1000=bootstrap_regression(drivers_SON_xr, drivers_SON_xr, 'IOD', ['ENSO'], n_boot=10000, sample_size=200, 
                                      add_intercept=False
                                      )
results_SPV_1000=bootstrap_regression(drivers_SON_xr, drivers_SON_xr, 'SPV', ['ENSO', 'IOD'], n_boot=10000, sample_size=200, 
                                      add_intercept=False
                                      )
results_A_SAM_1000=bootstrap_regression(drivers_SON_xr, drivers_SON_xr, 'A_SAM', ['ENSO', 'IOD'], n_boot=10000, sample_size=200, 
                                      add_intercept=False
                                      )
results_S_SAM_1000=bootstrap_regression(drivers_SON_xr, drivers_SON_xr, 'S_SAM', ['ENSO','IOD', 'SPV'], n_boot=10000, sample_size=200, 
                                      add_intercept=False
                                      )

plot_bootstrap_coefficients(results_IOD_1000, ['ENSO'], title='Bootstrap regression coefficients \n IOD',
                            era5_vals={'ENSO':0.55})
plot_bootstrap_coefficients(results_SPV_1000, ['ENSO', 'IOD'], title='Bootstrap regression coefficients \n SPV',
                            era5_vals={'ENSO':-0.08, 'IOD':0.05})
plot_bootstrap_coefficients(results_A_SAM_1000, ['ENSO', 'IOD'], title='Bootstrap regression coefficients \n A_SAM',
                            era5_vals={'ENSO':-0.57, 'IOD':-0.25})
plot_bootstrap_coefficients(results_S_SAM_1000, ['ENSO', 'IOD', 'SPV'], title='Bootstrap regression coefficients \n S_SAM',
                            era5_vals={'ENSO':-0.05, 'IOD':-0.08, 'SPV':0.72})


print('Finished SON mixing all!')

drivers_DJF=['ENSO', 'IOBW','VB', 'S_SAM', 'A_SAM']
drivers_DJF_tot=['ENSO', 'IOBW', 'VB']

#DJF direct effects on T and precip in both regions
era5_temp_Andes_direct_DJF={'ENSO':-0.08, 'IOBW':0.33, 'VB':-0.15, 'S_SAM':0.24, 'A_SAM':0.16}
era5_precip_Andes_direct_DJF={'ENSO':-0.28, 'IOBW':-0.04, 'VB':-0.14, 'S_SAM':-0.39, 'A_SAM':-0.04}
era5_temp_LP_direct_DJF={'ENSO':-0.25, 'IOBW':0.40, 'VB':-0.14, 'S_SAM':0.11, 'A_SAM':-0.25}
era5_precip_LP_direct_DJF={'ENSO':0.48, 'IOBW':-0.29, 'VB':-0.12, 'S_SAM':-0.05, 'A_SAM':0.08}

era5_val_direct_list=[era5_temp_Andes_direct_DJF, era5_precip_Andes_direct_DJF,
                       era5_temp_LP_direct_DJF, era5_precip_LP_direct_DJF]
heading_list=['Andes T','Andes Pr','La Plata T','La Plata Pr']
#direct effects on T and precip in both regions
results_direct_T_Andes_DJF_1000=bootstrap_regression(target_Andes_DJF_xr, drivers_DJF_xr, 't2m', drivers_DJF, n_boot=10000, sample_size=200, add_intercept=False)
results_direct_Pr_Andes_DJF_1000=bootstrap_regression(target_Andes_DJF_xr, drivers_DJF_xr, 'tp', drivers_DJF, n_boot=10000, sample_size=200, add_intercept=False)
results_direct_T_LP_DJF_1000=bootstrap_regression(target_LP_DJF_xr, drivers_DJF_xr, 't2m', drivers_DJF, n_boot=10000, sample_size=200, add_intercept=False)
results_direct_Pr_LP_DJF_1000=bootstrap_regression(target_LP_DJF_xr, drivers_DJF_xr, 'tp', drivers_DJF, n_boot=10000, sample_size=200, add_intercept=False)

for i, element in enumerate([results_direct_T_Andes_DJF_1000, results_direct_Pr_Andes_DJF_1000, 
                             results_direct_T_LP_DJF_1000, results_direct_Pr_LP_DJF_1000]):
    plot_bootstrap_coefficients(element, drivers_DJF,
                                title=f'Bootstrap regression coefficients \n {heading_list[i]}', 
                                era5_vals=era5_val_direct_list[i])

#total effects on T and precip in both regions
era5_temp_Andes_total_DJF={'ENSO':-0.17, 'IOBW':0.36, 'VB':-0.05}
era5_precip_Andes_total_DJF={'ENSO':-0.19, 'IOBW':-0.17, 'VB':-0.32}
era5_temp_LP_total_DJF={'ENSO':-0.24, 'IOBW':0.48, 'VB':-0.09}
era5_precip_LP_total_DJF={'ENSO':0.46, 'IOBW':-0.33, 'VB':-0.14}

era5_val_total_list=[era5_temp_Andes_total_DJF, era5_precip_Andes_total_DJF,
                     era5_temp_LP_total_DJF, era5_precip_LP_total_DJF]
tot_drivers_DJF=['ENSO', 'IOBW', 'VB']

results_tot_T_Andes_DJF_1000=bootstrap_regression(target_Andes_DJF_xr, drivers_DJF_xr, 't2m', tot_drivers_DJF, n_boot=10000, sample_size=200, add_intercept=False, total_eff=True)
results_tot_Pr_Andes_DJF_1000=bootstrap_regression(target_Andes_DJF_xr, drivers_DJF_xr, 'tp', tot_drivers_DJF, n_boot=10000, sample_size=200, add_intercept=False, total_eff=True)
results_tot_T_LP_DJF_1000=bootstrap_regression(target_LP_DJF_xr, drivers_DJF_xr, 't2m', tot_drivers_DJF, n_boot=10000, sample_size=200, add_intercept=False, total_eff=True)
results_tot_Pr_LP_DJF_1000=bootstrap_regression(target_LP_DJF_xr, drivers_DJF_xr, 'tp', tot_drivers_DJF, n_boot=10000, sample_size=200, add_intercept=False, total_eff=True)

for i, element in enumerate([results_tot_T_Andes_DJF_1000, results_tot_Pr_Andes_DJF_1000,
                             results_tot_T_LP_DJF_1000, results_tot_Pr_LP_DJF_1000]):
    plot_bootstrap_coefficients(element, tot_drivers_DJF, 
                                title=f'Bootstrap regression coefficients \n Total effects '+ f' {heading_list[i]}',
                                era5_vals=era5_val_total_list[i])


#other DJF links

#IOBW
era5_IOBW={'ENSO':0.81}
IOBW_DJF_1000=bootstrap_regression(drivers_DJF_xr, drivers_DJF_xr, 'IOBW', ['ENSO'], n_boot=10000,
                                    sample_size=200, add_intercept=False)
plot_bootstrap_coefficients(IOBW_DJF_1000, ['ENSO'], title='Bootstrap reg coef for IOBW DJF \n Init month: 9 + 10',
                            era5_vals=era5_IOBW)

IOBW_DJF_10000=bootstrap_regression(drivers_DJF_xr_no_VB, drivers_DJF_xr_no_VB, 'IOBW', ['ENSO'], n_boot=10000,
                                    sample_size=200, add_intercept=False)
plot_bootstrap_coefficients(IOBW_DJF_10000, ['ENSO'], title='Bootstrap reg coef for IOBW DJF \n All inits',
                            era5_vals=era5_IOBW) 
#A_SAM
era5_A_SAM_DJF={'ENSO':-0.44, 'IOBW':0.04}

A_SAM_DJF_init9=bootstrap_regression(drivers_DJF_xr, drivers_DJF_xr, 'A_SAM', ['ENSO', 'IOBW'], n_boot=10000,
                                    sample_size=200, add_intercept=False)
plot_bootstrap_coefficients(A_SAM_DJF_init9, ['ENSO', 'IOBW'], title='Bootstrap reg coef for A-SAM DJF \n Init month: 9 +10',
                            era5_vals=era5_A_SAM_DJF)

A_SAM_DJF_10000=bootstrap_regression(drivers_DJF_xr_no_VB, drivers_DJF_xr_no_VB, 'A_SAM', ['ENSO', 'IOBW'], n_boot=10000,
                                    sample_size=200, add_intercept=False)
plot_bootstrap_coefficients(A_SAM_DJF_10000, ['ENSO', 'IOBW'], title='Bootstrap reg coef for A-SAM DJF \n All inits',
                            era5_vals=era5_A_SAM_DJF)
#VB
era5_VB_DJF={'ENSO':-0.23,'IOBW':0.26}
VB_DJF_init9=bootstrap_regression(drivers_DJF_xr, drivers_DJF_xr, 'VB', ['ENSO', 'IOBW'], n_boot=10000,
                                    sample_size=200, add_intercept=False)
plot_bootstrap_coefficients(VB_DJF_init9, ['ENSO', 'IOBW'], title='Bootstrap reg coef for VB DJF \n Init month: 9 +10', era5_vals=era5_VB_DJF)

#S_SAM
era5_S_SAM_DJF={'ENSO':-0.18, 'IOBW':0.1, 'VB':0.44}
S_SAM_DJF_init9=bootstrap_regression(drivers_DJF_xr, drivers_DJF_xr, 'S_SAM', ['ENSO', 'IOBW', 'VB'], n_boot=10000,
                                    sample_size=200, add_intercept=False)
plot_bootstrap_coefficients(S_SAM_DJF_init9, ['ENSO', 'IOBW', 'VB'], title='Bootstrap reg coef for S-SAM DJF \n Init month: 9 +10', era5_vals=era5_S_SAM_DJF)

print('Finished DJF mixing all!')

#Start overview Figure
spring_driver_links=[results_IOD_1000['coef_ENSO'].mean(), *[
        results_SPV_1000[f'coef_{var}'].mean()
        for var in ['ENSO', 'IOD']
    ], *[
        results_A_SAM_1000[f'coef_{var}'].mean()
        for var in ['ENSO', 'IOD']
    ], *[
        results_S_SAM_1000[f'coef_{var}'].mean()
        for var in ['ENSO', 'IOD', 'SPV']
    ]]

summer_driver_links=[IOBW_DJF_1000['coef_ENSO'].mean(),
                     *[VB_DJF_init9[f'coef_{var}'].mean() for var in ['ENSO', 'IOBW']],
                     *[A_SAM_DJF_init9[f'coef_{var}'].mean() for var in ['ENSO', 'IOBW']],
                      *[S_SAM_DJF_init9[f'coef_{var}'].mean() for var in ['ENSO', 'IOBW', 'VB']]]
print(spring_driver_links)

def transform_reg_lists(
    somelist,
    driver_target_tuple_list,
    dict_with_other_info=None,
    copy_ind=True
):

    if copy_ind:
        d = (
            {}
            if dict_with_other_info is None
            else dict_with_other_info.copy()
        )
    else:
        d = (
            {}
            if dict_with_other_info is None
            else dict_with_other_info
        )

    # Flatten nested list if necessary
    if driver_target_tuple_list and isinstance(
        driver_target_tuple_list[0], list
    ):
        keys = [
            key
            for group in driver_target_tuple_list
            for key in group
        ]
    else:
        keys = driver_target_tuple_list

    if len(somelist) != len(keys):
        raise ValueError(
            f"Number of coefficients ({len(somelist)}) "
            f"does not match number of driver-target pairs "
            f"({len(keys)})"
        )

    for reg_coef, key in zip(somelist, keys):
        d[key] = float(reg_coef)

    return d

driver_target_list_SON=[[('ENSO', 'IOD')], [('ENSO', 'SPV'), ('IOD', 'SPV')], [('ENSO', 'A_SAM'), ('IOD', 'A_SAM')],
                    [('ENSO', 'S_SAM'), ('IOD', 'S_SAM'), ('SPV', 'S_SAM')]]
driver_target_list_DJF=[[('ENSO', 'IOBW')], [('ENSO', 'VB'), ('IOBW', 'VB')], [('ENSO', 'A_SAM'), ('IOBW', 'A_SAM')],
                    [('ENSO', 'S_SAM'), ('IOBW', 'S_SAM'), ('VB', 'S_SAM')]]

mod_dict_SON = {}
mod_dict_DJF = {}



mod_dict_SON = transform_reg_lists(
        spring_driver_links,
        driver_target_list_SON,
        mod_dict_SON, copy_ind=False
    )

mod_dict_DJF = transform_reg_lists(
         summer_driver_links,
         driver_target_list_DJF,
         mod_dict_DJF, copy_ind=False
     )

def mean_from_df(df, drivers_list=drivers_SON):
    mean_list=[]
    for var in drivers_list:
        mean=df[f'coef_{var}'].mean()
        mean_list.append(mean)
    return mean_list

target_vars=['T_Andes', 'Precip_Andes', 'T_LP', 'Precip_LP']
target_reg_coef_list_SON=[mean_from_df(results_direct_T_Andes_SON_1000), mean_from_df(results_direct_Pr_Andes_SON_1000), 
                          mean_from_df(results_direct_T_LP_SON_1000), mean_from_df(results_direct_Pr_LP_SON_1000)]
target_reg_coef_list_DJF=[mean_from_df(results_direct_T_Andes_DJF_1000, drivers_DJF), mean_from_df(results_direct_Pr_Andes_DJF_1000, drivers_DJF),
                           mean_from_df(results_direct_T_LP_DJF_1000, drivers_DJF), mean_from_df(results_direct_Pr_LP_DJF_1000, drivers_DJF)]

pos_t_SON = { "ENSO": (0, 2), "SPV": (1.2, 1), 'IOD':(-1.2,1), "S_SAM": (+0.5, .5), "A_SAM": (-0.5, 0.5), 
             "T": (0, -1)} 
pos_precip_SON = { "ENSO": (0, 2), "SPV": (1.2, 1), 'IOD':(-1.2,1), "S_SAM": (+0.5, .5),
                   "A_SAM": (-0.5, 0.5), "Precip": (0, -1)} 
pos_t_DJF = { "ENSO": (0, 2), "VB": (1.2, 1), 'IOBW':(-1.2,1), "S_SAM": (+0.5, .5), 
             "A_SAM": (-0.5, 0.5), "T": (0, -1)}
pos_precip_DJF = { "ENSO": (0, 2), "VB": (1.2, 1), 'IOBW':(-1.2,1), "S_SAM": (+0.5, .5),
                   "A_SAM": (-0.5, 0.5), "Precip": (0, -1)}

d_SON={}
d_DJF={}
for i in range(len(target_vars)):
    target_tuple_list=[(driver, target_vars[i]) for driver in drivers_SON]
    target_tuple_list_DJF=[(driver, target_vars[i]) for driver in drivers_DJF]
    SON_target_dict=transform_reg_lists(target_reg_coef_list_SON[i], target_tuple_list, mod_dict_SON)
    DJF_target_dict=transform_reg_lists(target_reg_coef_list_DJF[i], target_tuple_list_DJF, mod_dict_DJF)
    #print(SON_target_dict)
    d_SON[target_vars[i]]=SON_target_dict
    d_DJF[target_vars[i]]= DJF_target_dict


var_type = target_vars[i].split("_")[0]
positions_list = []

for i, var in enumerate(target_vars):
    var_type = var.split("_")[0]   # T or Precip

    # SON block
    if var_type == "T":
        positions_list.append(pos_t_SON)
    else:
        positions_list.append(pos_precip_SON)

# DJF block (same order appended after SON)
for i, var in enumerate(target_vars):
     var_type = var.split("_")[0]

     if var_type == "T":
         positions_list.append(pos_t_DJF)
     else:
         positions_list.append(pos_precip_DJF)



network_data_list=list(d_SON.values())+list(d_DJF.values())
title_list = []

for var in target_vars:
    region = var.split("_")[1]

    title_list.append(f"{var.split('_')[0]} {region}")   # SON
for var in target_vars:
    region = var.split("_")[1]

    title_list.append(f"{var.split('_')[0]} {region}")   # DJF

    
positions_list=[pos_t_SON, pos_precip_SON, pos_t_SON, pos_precip_SON,
                 pos_t_DJF, pos_precip_DJF, pos_t_DJF, pos_precip_DJF]


print(network_data_list)

plot_causal_networks_grid(network_data_list, positions_list, title_list, row_labels=['SON', 'DJF'], 
                          heading_add='Hindcast mean coefficients')

print('Finished overview figure!')

###################################################################################################
#2. Stratify by init month
###################################################################################################

#SON

for init_month in np.unique(target_Andes_SON_xr.init_month.values):
    print(f"Processing SON init month: {init_month}")
    target_Andes_SON_xr_init = target_Andes_SON_xr.where(target_Andes_SON_xr.init_month == init_month, drop=True)
    target_LP_SON_xr_init = target_LP_SON_xr.where(target_LP_SON_xr.init_month == init_month, drop=True)
    drivers_aligned_init = drivers_aligned.where(drivers_aligned.init_month == init_month, drop=True)
    drivers_SON_xr_init = xr.merge(drivers_aligned_init)

    # Perform regression analysis for this specific init month

    #direct effects
    results_direct_T_Andes_SON_init = bootstrap_regression(target_Andes_SON_xr_init, drivers_SON_xr_init, 't2m', 
                                                           drivers_SON, n_boot=10000, sample_size=200, add_intercept=False)
    results_direct_Pr_Andes_SON_init = bootstrap_regression(target_Andes_SON_xr_init, 
                                                            drivers_SON_xr_init, 'tp', drivers_SON, 
                                                            n_boot=10000, sample_size=200, add_intercept=False)
    results_direct_T_LP_SON_init = bootstrap_regression(target_LP_SON_xr_init, drivers_SON_xr_init, 't2m', 
                                                        drivers_SON, n_boot=10000, sample_size=200, add_intercept=False)
    results_direct_Pr_LP_SON_init = bootstrap_regression(target_LP_SON_xr_init, drivers_SON_xr_init, 'tp', 
                                                         drivers_SON, n_boot=10000, sample_size=200, add_intercept=False)

    # Plotting the results for this specific init month
    plot_bootstrap_coefficients(results_direct_T_Andes_SON_init, drivers_SON,
                                title=f'Bootstrap regression coefficients \n Andes T SON for Init Month {init_month}')
    plot_bootstrap_coefficients(results_direct_Pr_Andes_SON_init, drivers_SON,
                                title=f'Bootstrap regression coefficients \n Andes Pr SON for Init Month {init_month}')
    plot_bootstrap_coefficients(results_direct_T_LP_SON_init, drivers_SON,
                                title=f'Bootstrap regression coefficients \n La Plata T SON for Init Month {init_month}')   
    plot_bootstrap_coefficients(results_direct_Pr_LP_SON_init, drivers_SON,
                                title=f'Bootstrap regression coefficients \n La Plata Pr SON for Init Month {init_month}')

    #total effects
    results_tot_T_Andes_SON_init = bootstrap_regression(target_Andes_SON_xr_init, drivers_SON_xr_init, 't2m', tot_drivers, 
                                                        n_boot=10000, sample_size=200, add_intercept=False, total_eff=True)
    results_tot_Pr_Andes_SON_init = bootstrap_regression(target_Andes_SON_xr_init, drivers_SON_xr_init, 'tp', tot_drivers, 
                                                         n_boot=10000, sample_size=200, add_intercept=False, total_eff=True)
    results_tot_T_LP_SON_init = bootstrap_regression(target_LP_SON_xr_init, drivers_SON_xr_init, 't2m', tot_drivers,
                                                     n_boot=10000, sample_size=200, add_intercept=False, total_eff=True)
    results_tot_Pr_LP_SON_init = bootstrap_regression(target_LP_SON_xr_init, drivers_SON_xr_init, 'tp', tot_drivers, 
                                                      n_boot=10000, sample_size=200, add_intercept=False, total_eff=True)
    
    # Plotting the total effects results for this specific init month
    plot_bootstrap_coefficients(results_tot_T_Andes_SON_init, tot_drivers,
                                title=f'Bootstrap regression coefficients \n Total Effects Andes T SON for Init Month {init_month}')
    plot_bootstrap_coefficients(results_tot_Pr_Andes_SON_init, tot_drivers,
                                title=f'Bootstrap regression coefficients \n Total Effects Andes Pr SON for Init Month {init_month}')
    plot_bootstrap_coefficients(results_tot_T_LP_SON_init, tot_drivers,
                                title=f'Bootstrap regression coefficients \n Total Effects La Plata T SON for Init Month {init_month}')   
    plot_bootstrap_coefficients(results_tot_Pr_LP_SON_init, tot_drivers,
                                title=f'Bootstrap regression coefficients \n Total Effects La Plata Pr SON for Init Month {init_month}')


    #other links for this specific init month
    results_IOD_init = bootstrap_regression(drivers_SON_xr_init, drivers_SON_xr_init, 'IOD', ['ENSO'], n_boot=10000, sample_size=200, add_intercept=False)
    results_SPV_init = bootstrap_regression(drivers_SON_xr_init, drivers_SON_xr_init, 'SPV', ['ENSO', 'IOD'], n_boot=10000, sample_size=200, add_intercept=False)
    results_A_SAM_init = bootstrap_regression(drivers_SON_xr_init, drivers_SON_xr_init, 'A_SAM', ['ENSO', 'IOD'], n_boot=10000, sample_size=200, add_intercept=False)
    results_S_SAM_init = bootstrap_regression(drivers_SON_xr_init, drivers_SON_xr_init, 'S_SAM', ['ENSO','IOD', 'SPV'], n_boot=10000, sample_size=200, add_intercept=False)

    plot_bootstrap_coefficients(results_IOD_init, ['ENSO'], title=f'Bootstrap regression coefficients \n IOD SON for Init Month {init_month}')
    plot_bootstrap_coefficients(results_SPV_init, ['ENSO', 'IOD'], title=f'Bootstrap regression coefficients \n SPV SON for Init Month {init_month}')
    plot_bootstrap_coefficients(results_A_SAM_init, ['ENSO', 'IOD'], title=f'Bootstrap regression coefficients \n A_SAM SON for Init Month {init_month}')
    plot_bootstrap_coefficients(results_S_SAM_init, ['ENSO', 'IOD', 'SPV'], title=f'Bootstrap regression coefficients \n S_SAM SON for Init Month {init_month}')

print('Finished SON stratified by init month!')

for init_month in np.unique(target_Andes_DJF_xr.init_month.values):
    print(f"Processing DJF init month: {init_month}")
    target_Andes_DJF_xr_init = target_Andes_DJF_xr.where(target_Andes_DJF_xr.init_month == init_month, drop=True)
    target_LP_DJF_xr_init = target_LP_DJF_xr.where(target_LP_DJF_xr.init_month == init_month, drop=True)
    drivers_aligned_init = drivers_aligned.where(drivers_aligned.init_month == init_month, drop=True)
    drivers_DJF_xr_init = xr.merge(drivers_aligned_init)

    # Perform regression analysis for this specific init month

    #direct effects
    results_direct_T_Andes_DJF_init = bootstrap_regression(target_Andes_DJF_xr_init, drivers_DJF_xr_init, 't2m', 
                                                           drivers_DJF, n_boot=10000, sample_size=200, add_intercept=False)
    results_direct_Pr_Andes_DJF_init = bootstrap_regression(target_Andes_DJF_xr_init, 
                                                            drivers_DJF_xr_init, 'tp', drivers_DJF, 
                                                            n_boot=10000, sample_size=200, add_intercept=False)
    results_direct_T_LP_DJF_init = bootstrap_regression(target_LP_DJF_xr_init, drivers_DJF_xr_init, 't2m', 
                                                        drivers_DJF, n_boot=10000, sample_size=200, add_intercept=False)
    results_direct_Pr_LP_DJF_init = bootstrap_regression(target_LP_DJF_xr_init, drivers_DJF_xr_init, 'tp', 
                                                         drivers_DJF, n_boot=10000, sample_size=200, add_intercept=False)

    # Plotting the results for this specific init month
    plot_bootstrap_coefficients(results_direct_T_Andes_DJF_init, drivers_DJF,
                                title=f'Bootstrap regression coefficients \n Andes T DJF for Init Month {init_month}')
    plot_bootstrap_coefficients(results_direct_Pr_Andes_DJF_init, drivers_DJF,
                                title=f'Bootstrap regression coefficients \n Andes Pr DJF for Init Month {init_month}')
    plot_bootstrap_coefficients(results_direct_T_LP_DJF_init, drivers_DJF,
                                title=f'Bootstrap regression coefficients \n La Plata T DJF for Init Month {init_month}')
    plot_bootstrap_coefficients(results_direct_Pr_LP_DJF_init, drivers_DJF,
                                title=f'Bootstrap regression coefficients \n La Plata Pr DJF for Init Month {init_month}')

    #total effects
    results_tot_T_Andes_DJF_init = bootstrap_regression(target_Andes_DJF_xr_init, drivers_DJF_xr_init, 't2m', tot_drivers_DJF, 
                                                        n_boot=10000, sample_size=200, add_intercept=False, total_eff=True)
    results_tot_Pr_Andes_DJF_init = bootstrap_regression(target_Andes_DJF_xr_init, drivers_DJF_xr_init, 'tp', tot_drivers_DJF, 
                                                         n_boot=10000, sample_size=200, add_intercept=False, total_eff=True)
    results_tot_T_LP_DJF_init = bootstrap_regression(target_LP_DJF_xr_init, drivers_DJF_xr_init, 't2m', tot_drivers_DJF,
                                                     n_boot=10000, sample_size=200, add_intercept=False, total_eff=True)
    results_tot_Pr_LP_DJF_init = bootstrap_regression(target_LP_DJF_xr_init, drivers_DJF_xr_init, 'tp', tot_drivers_DJF, 
                                                      n_boot=10000, sample_size=200, add_intercept=False, total_eff=True)

    # Plotting the total effects results for this specific init month
    plot_bootstrap_coefficients(results_tot_T_Andes_DJF_init, tot_drivers,
                                title=f'Bootstrap regression coefficients \n Total Effects Andes T DJF for Init Month {init_month}')
    plot_bootstrap_coefficients(results_tot_Pr_Andes_DJF_init, tot_drivers,
                                title=f'Bootstrap regression coefficients \n Total Effects Andes Pr DJF for Init Month {init_month}')
    plot_bootstrap_coefficients(results_tot_T_LP_DJF_init, tot_drivers,
                                title=f'Bootstrap regression coefficients \n Total Effects La Plata T DJF for Init Month {init_month}')
    plot_bootstrap_coefficients(results_tot_Pr_LP_DJF_init, tot_drivers,
                                title=f'Bootstrap regression coefficients \n Total Effects La Plata Pr DJF for Init Month {init_month}')

    #other links for this specific init month
    results_IOBW_init = bootstrap_regression(drivers_DJF_xr_init, drivers_DJF_xr_init, 'IOBW', ['ENSO'], n_boot=10000, sample_size=200, add_intercept=False)
    results_VB_init = bootstrap_regression(drivers_DJF_xr_init, drivers_DJF_xr_init, 'VB', ['ENSO', 'IOBW'], n_boot=10000, sample_size=200, add_intercept=False)
    results_A_SAM_init = bootstrap_regression(drivers_DJF_xr_init, drivers_DJF_xr_init, 'A_SAM', ['ENSO', 'IOBW'], n_boot=10000, sample_size=200, add_intercept=False)
    results_S_SAM_init = bootstrap_regression(drivers_DJF_xr_init, drivers_DJF_xr_init, 'S_SAM', ['ENSO','IOBW', 'VB'], n_boot=10000, sample_size=200, add_intercept=False)

    plot_bootstrap_coefficients(results_IOBW_init, ['ENSO'], title=f'Bootstrap regression coefficients \n IOBW DJF for Init Month {init_month}')
    plot_bootstrap_coefficients(results_VB_init, ['ENSO', 'IOBW'], title=f'Bootstrap regression coefficients \n VB DJF for Init Month {init_month}')
    plot_bootstrap_coefficients(results_A_SAM_init, ['ENSO', 'IOBW'], title=f'Bootstrap regression coefficients \n A_SAM DJF for Init Month {init_month}')
    plot_bootstrap_coefficients(results_S_SAM_init, ['ENSO', 'IOBW', 'VB'], title=f'Bootstrap regression coefficients \n S_SAM DJF for Init Month {init_month}')

    #overview figure for this specific init month
    spring_driver_links_init=[results_IOD_init['coef_ENSO'].mean(), *[
            results_SPV_init[f'coef_{var}'].mean()
            for var in ['ENSO', 'IOD']
        ], *[
            results_A_SAM_init[f'coef_{var}'].mean()
            for var in ['ENSO', 'IOD']
        ], *[
            results_S_SAM_init[f'coef_{var}'].mean()
            for var in ['ENSO', 'IOD', 'SPV']
        ]]  

    summer_driver_links_init=[results_IOBW_init['coef_ENSO'].mean(),
                             *[results_VB_init[f'coef_{var}'].mean() for var in ['ENSO', 'IOBW']],
                             *[results_A_SAM_init[f'coef_{var}'].mean() for var in ['ENSO', 'IOBW']],
                              *[results_S_SAM_init[f'coef_{var}'].mean() for var in ['ENSO', 'IOBW', 'VB']]]

    mod_dict_SON_init = transform_reg_lists(
            spring_driver_links_init,
            driver_target_list_SON,
            mod_dict_SON, copy_ind=False
        )
    mod_dict_DJF_init = transform_reg_lists(
             summer_driver_links_init,
             driver_target_list_DJF,
             mod_dict_DJF, copy_ind=False
         )

    target_reg_coef_list_SON_init=[mean_from_df(results_direct_T_Andes_SON_init), mean_from_df(results_direct_Pr_Andes_SON_init),
                                  mean_from_df(results_direct_T_LP_SON_init), mean_from_df(results_direct_Pr_LP_SON_init)]
    target_reg_coef_list_DJF_init=[mean_from_df(results_direct_T_Andes_DJF_init, drivers_DJF), mean_from_df(results_direct_Pr_Andes_DJF_init, drivers_DJF),
                                   mean_from_df(results_direct_T_LP_DJF_init, drivers_DJF), mean_from_df(results_direct_Pr_LP_DJF_init, drivers_DJF)]

    d_SON_init={}
    d_DJF_init={}
    for i in range(len(target_vars)):
        target_tuple_list=[(driver, target_vars[i]) for driver in drivers_SON]
        target_tuple_list_DJF=[(driver, target_vars[i]) for driver in drivers_DJF]
        SON_target_dict_init=transform_reg_lists(target_reg_coef_list_SON_init[i], target_tuple_list, mod_dict_SON_init)
        DJF_target_dict_init=transform_reg_lists(target_reg_coef_list_DJF_init[i], target_tuple_list_DJF, mod_dict_DJF_init)
        d_SON_init[target_vars[i]]=SON_target_dict_init
        d_DJF_init[target_vars[i]]= DJF_target_dict_init    

    network_data_list_init=list(d_SON_init.values())+list(d_DJF_init.values())
    plot_causal_networks_grid(network_data_list_init, positions_list, title_list, row_labels=['SON', 'DJF'], heading_add=f'Hindcast mean coefficients for Init Month {init_month}')

print('Finished DJF stratified by init month!')

print('Script completed')