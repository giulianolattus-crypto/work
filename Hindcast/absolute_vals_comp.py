import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA
import scipy.signal as signal
import statsmodels.api as sm
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def weight_by_latitude(da):
    """Weight data array by cosine of latitude."""
    weights = np.cos(np.deg2rad(da['latitude']))
    weighted_da = da * weights
    return weighted_da


def reshape_to_2d(da):
    """Reshape (lat, lon, sample) data into 2D matrix (space x sample)."""
    space_dim = da['latitude'].size * da['longitude'].size
    ##modify here likely!
    reshaped = da.stack(space=('latitude', 'longitude')).transpose('sample', 'space')
    return reshaped


def compute_covariance_matrix(reshaped_da):
    """Compute the covariance matrix of the data."""
    data = reshaped_da.values
    data = data - np.nanmean(data, axis=0)  # Centering the data
    covariance_matrix = np.cov(data.T)
    return covariance_matrix


def perform_svd(cov_matrix):
    """Perform SVD on the covariance matrix."""
    U, S, Vt = np.linalg.svd(cov_matrix)
    return U, S, Vt


def plot_first_principal_component(reshaped_da, Vt):
    """Plot the first principal component as a map."""
    first_pc = Vt[0, :] #first eigenvector with largest eigenvalue
    pc_map = reshaped_da.isel(time=0).copy() #.isel selects the i you give, make a copy of given array at i
    pc_map.values = first_pc #change the values from before to the first_pc values
    pc_map_unstacked = pc_map.unstack('space')

    plt.figure(figsize=(8, 5))
    pc_map_unstacked.plot(cmap='RdBu')
    plt.title("First Principal Component (Spatial Pattern)")
    plt.show()

    return pc_map_unstacked


def compute_zonal_mean(pc_map_unstacked):
    """Compute the zonal mean of the first principal component map."""
    zonal_mean = pc_map_unstacked.mean(dim='longitude')
    return zonal_mean


def make_zonal_mean_map(zonal_mean, da):
    """Create a zonal mean map, assigning each longitude the same value."""
    zonal_map = xr.zeros_like(da.isel(time=0)) #blank map with same dimensions as da
    for lon in zonal_map['longitude']:
        zonal_map.loc[{'longitude': lon}] = zonal_mean #assign zonal mean value to each longitude
    return zonal_map


def regress_onto_zonal_mean_map(da, zonal_map):
    """Regress original data onto the zonal mean map (time series of coefficients)."""
#flattening data
    zonal_map_flat = zonal_map.stack(space=('latitude', 'longitude')).values
    data_flat = da.stack(space=('latitude', 'longitude')).transpose('sample', 'space').values
#detrending by removing mean
    zonal_map_flat = zonal_map_flat - np.nanmean(zonal_map_flat)
    data_flat = data_flat - np.nanmean(data_flat, axis=0)

    regression_coefficients = np.dot(data_flat, zonal_map_flat) / np.dot(zonal_map_flat, zonal_map_flat)

    return regression_coefficients


def plot_regression_coefficients(time, regression_coefficients):
    """Plot regression coefficients over time."""
    plt.figure(figsize=(8, 5))
    plt.plot(time, regression_coefficients, label='Regression Coefficients')
    plt.xlabel('Time')
    plt.ylabel('Coefficient')
    plt.title('Regression Coefficients Over Time')
    plt.grid()
    plt.show()
        


def seasonal_average(da):
    da2=da.groupby('time.year').mean('time')
    return(da2)

def stdize_ssavg(da):
    da2=(da-da.mean())/da.std()
    return da2


##Open data
# Path to your NetCDF file
DATADIR='/climca/data/SEAS5_SA/data'
file_new = f"{DATADIR}/seas5_single_levels_06-12month_1-6leadtimemonth.nc"
# Open the dataset
# chunks is useful because your dataset is huge.
# It avoids loading everything into memory immediately.

ds=xr.open_dataset(file_new, chunks={
        "number": 1,
        "forecast_reference_time": 10,
        "forecastMonth": 6
    })


# ------------------------------------------------------------
# Make sure we have valid_time
# ------------------------------------------------------------
# valid_time = real calendar month of the forecast
#
# Example:
# forecast_reference_time = 1981-09-01
# forecastMonth = 1 -> valid_time = 1981-09-01
# forecastMonth = 2 -> valid_time = 1981-10-01
# forecastMonth = 3 -> valid_time = 1981-11-01
# forecastMonth = 4 -> valid_time = 1981-12-01
# forecastMonth = 5 -> valid_time = 1982-01-01
# forecastMonth = 6 -> valid_time = 1982-02-01

#For instance for a forecast with start date on the 1st May, forecastMonth=1 is May,
#  and for a forecast with start date on 17th April, forecastMonth=1 would also be May. 
# This is more coherent and avoids the ambiguity described above.
def make_valid_time(ds):
    if "valid_time" in ds.data_vars:
        # Sometimes valid_time exists but is stored as a data variable.
        # We promote it to a coordinate.
        ds = ds.set_coords("valid_time")


    elif "valid_time" not in ds.coords:
        # If valid_time does not exist, we create it manually.

        init_times = pd.to_datetime(ds["forecast_reference_time"].values)
        lead_months = ds["forecastMonth"].values

        valid_times = []

        for init in init_times:
            valid_times_this_init = []

            for lead in lead_months:
                this_valid_time = init + pd.DateOffset(months=int(lead) - 1)
                valid_times_this_init.append(this_valid_time)

            valid_times.append(valid_times_this_init)

        valid_times = np.array(valid_times, dtype="datetime64[ns]")

        ds = ds.assign_coords(
            valid_time=(
                ("forecast_reference_time", "forecastMonth"),
                valid_times
            )
        )

    print(ds["valid_time"])
    return ds

ds=make_valid_time(ds)


##Make blocks
def make_3_month_block(ds, lead_months, block_name):
    """
    Select a 3-month forecast block.

    lead_months can be:
        [1, 2, 3]
        [4, 5, 6]

    block_name is just a label:
        "lead_1_2_3"
        "lead_4_5_6"
    """

    #select requested lead months
    block = ds.sel(forecastMonth=lead_months)

    #save them in new coordinate
    block = block.assign_coords(
        original_forecastMonth=("forecastMonth", lead_months)
    )

    #rename old coordinate to new name
    block = block.rename({"forecastMonth": "season_month"})

    #reset month numbering
    block = block.assign_coords(
        season_month=[1, 2, 3]
    )
    #assign lead block name
    block = block.assign_coords(
        lead_block=block_name
    )

    return block

#Find lead times
def season_to_leads(init_month, season_months):
    """
    Convert calendar months to forecast lead months.

    Parameters
    ----------
    init_month : int
        Initialization month (1=Jan, ..., 12=Dec)

    season_months : list
        Calendar months, e.g. [9,10,11] for SON.

    Returns
    -------
    list
        Corresponding forecast lead months.
    """
    leads = []

    for month in season_months:
        lead = (month - init_month+1) % 12
        if lead == 0:
            lead = 12
        leads.append(lead)

    return leads

def make_3_month_block_old(ds, lead_months, init_month):
    """
    Select a 3-month forecast block.

    lead_months can be:
        [1, 2, 3]
        [4, 5, 6]

    block_name is just a label:
        "lead_1_2_3"
        "lead_4_5_6"
    """

    #select requested lead months
    block = ds.sel(forecastMonth=lead_months)

    #save them in new coordinate
    block = block.assign_coords(
        original_forecastMonth=("forecastMonth", lead_months)
    )

    #rename old coordinate to new name
    block = block.rename({"forecastMonth": "season_month"})

    #reset month numbering
    block = block.assign_coords(
        season_month=[1, 2, 3]
    )
    #assign lead block name
    block = block.assign_coords(
        init_month=init_month
    )

    return block

def make_3_month_block(ds, lead_months, init_month, block_name):

    # Select forecast initializations occurring in this month
    mask = ds.forecast_reference_time.dt.month == init_month

    block = ds.where(mask, drop=True)

    # Select the required forecast leads
    block = block.sel(
        forecastMonth=lead_months
    )

    # Save original lead numbers
    block = block.assign_coords(
        original_forecastMonth=(
            "forecastMonth",
            lead_months
        )
    )

    # Rename lead dimension
    block = block.rename(
        {"forecastMonth": "season_month"}
    )

    # Seasonal positions
    block = block.assign_coords(
        season_month=[1, 2, 3]
    )

    # Create an actual init_month dimension
    block = block.expand_dims(
        init_month=[init_month]
    )

    # Season label
    block = block.assign_coords(
        lead_block=block_name
    )

    return block

spring_leads_list=[]
for init_month in [6,7,8,9]:
    ##spring season block
    lead_months_spring=season_to_leads(init_month=init_month, season_months=[9,10,11])
    print(lead_months_spring)
    block_spring=make_3_month_block(ds, lead_months=lead_months_spring, init_month=init_month, block_name='SON_block')
    spring_leads_list.append(block_spring)

summer_leads_list=[]
for init_month in [9,10,11,12]:
    ##summer season block
    lead_months_summer=season_to_leads(init_month=init_month, season_months=[12,1,2])
    print(lead_months_summer)
    block_summer=make_3_month_block(ds, lead_months=lead_months_summer, init_month=init_month, block_name='DJF_block')
    summer_leads_list.append(block_summer)


print(block_spring)

#Combine each block
ds_3m_SON = xr.concat(
    spring_leads_list,
    dim="init_month")

ds_3m_DJF = xr.concat(
    summer_leads_list,
    dim="init_month")



#one sample consists of one ensemble member, one initialization date, one lead block 
#e.g. sample #1234: member 17, initialized 1981-09-01, lead_4_5_6 
# -->season_month=1 -> Dec 1981
# season_month=2 -> Jan 1982
# season_month=3 -> Feb 1982
def create_sample_and_cobmine_seasons(ds_3m_SON, ds_3m_DJF):

    #add season coordinate
    ds_3m_SON = ds_3m_SON.expand_dims(
    season=["SON"]
    )

    ds_3m_DJF = ds_3m_DJF.expand_dims(
        season=["DJF"]
    )

    #create sample multiindex
    ds_3m_SON = ds_3m_SON.stack(
        sample=("init_month", 'season', "forecast_reference_time", "number")
    )
    ds_3m_DJF = ds_3m_DJF.stack(
        sample=("init_month", 'season', "forecast_reference_time", "number")
    )

    

    ds_3m_SON = ds_3m_SON.transpose(
        "sample",
        "season_month",
        "latitude",
        "longitude"
    )

    ds_3m_DJF = ds_3m_DJF.transpose(
        "sample",
        "season_month",
        "latitude",
        "longitude"
    )
    return ds_3m_SON, ds_3m_DJF

ds_3m_list = create_sample_and_cobmine_seasons(ds_3m_SON, ds_3m_DJF)


SON = ds_3m_list[0].where(ds_3m_list[0]["season"] == "SON", drop=True)

DJF = ds_3m_list[1].where(ds_3m_list[1]["season"] == "DJF", drop=True)


print("SON samples:", SON.sizes["sample"])
print("DJF samples:", DJF.sizes["sample"])

SON_mean = SON.mean(dim="season_month")
DJF_mean = DJF.mean(dim="season_month")


###############################################################
#DATA PREPARATION
#################################################################
print('Data preparation start')
def select_region(ds, lat_min, lat_max, lon_min, lon_max):
    """
    Select a latitude-longitude box.

    Works even if latitude is ordered north-to-south, as in ERA5/SEAS5:
        90, 89, 88, ..., -90

    Longitude in your dataset is 0 to 360.
    So:
        -76 to -70 becomes 284 to 290
        -62 to -49 becomes 298 to 311
    """

    lat = ds["latitude"]

    # If latitude is decreasing: 90, 89, ..., -90
    if lat[0] > lat[-1]:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)

    out = ds.sel(
        latitude=lat_slice,
        longitude=slice(lon_min, lon_max)
    )

    return out

#not necessary here
def compute_sample_anomaly(ds):
    """
    Compute anomaly across all samples.

    For each grid point:
        anomaly = value - mean over samples

    This keeps all ensemble members and all years.
    """

    return ds - ds.mean(dim="sample")

#also not necessary here?
def detrend_across_time(ds):
    """
    Remove a linear trend across time.

    Important:
    Your dimension is called 'sample', not 'time'.

    Each sample has a coordinate called 'forecast_reference_time'.
    We extract the year from that coordinate and regress each grid point
    against year.

    Because you have 51 ensemble members per year, the same year appears
    many times. That is okay: the trend is fitted across all samples.
    """

    # Extract year from forecast_reference_time
    years = xr.DataArray(
        ds["forecast_reference_time"].dt.year.values,
        dims="sample",
        coords={"sample": ds["sample"]},
        name="year"
    )

    # Center the year to make the regression numerically cleaner
    x = years - years.mean(dim="sample")

    # Output container
    detrended_vars = {}

    for var in ds.data_vars:

        y = ds[var]

        # Mean over sample
        y_mean = y.mean(dim="sample")

        # Linear regression slope:
        # slope = cov(x, y) / var(x)
        slope = ((x * (y - y_mean)).mean(dim="sample")) / ((x ** 2).mean(dim="sample"))

        # Fitted linear trend
        fit = y_mean + slope * x

        # Remove fitted trend
        detrended_vars[var] = y - fit

    out = xr.Dataset(detrended_vars, coords=ds.coords)

    return out


def spatial_average(ds):
    """
    Latitude-weighted spatial average.

    Output:
        one value per sample
    """

    weights = np.cos(np.deg2rad(ds["latitude"]))

    out = ds.weighted(weights).mean(dim=["latitude", "longitude"])

    return out

#also not necessary here
def standardize(ds):
    """
    Standardize each variable across samples.

    z = value / std

    The mean should already be near zero after anomaly + detrending.
    If you want strict z-score, use:
        (ds - ds.mean("sample")) / ds.std("sample")
    """

    return ds / ds.std(dim="sample")

def process_region(ds_season, lat_min, lat_max, lon_min, lon_max):
    """
    Full pipeline for one region and one season.

    Input:
        ds_season = SON_mean or DJF_mean

    Output:
        standardized anomaly time series for all samples
    """

    # 1. Select region
    region = select_region(
        ds_season,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max
    )

    # 2. Anomaly over samples
    region_an = compute_sample_anomaly(region)

    # 3. Remove linear trend across time/sample
    region_detr = detrend_across_time(region_an)

    # 4. Spatial average
    region_avg = spatial_average(region_detr)

    # 5. Standardize
    region_std = standardize(region_avg)

    return region_std


# Andes
# Original longitude: -76 to -70
# In 0..360: 284 to 290

ANDES = {
    "lat_min": -55,
    "lat_max": -35,
    "lon_min": 284,
    "lon_max": 290,
}


# La Plata
# Original longitude: -62 to -49
# In 0..360: 298 to 311

LA_PLATA = {
    "lat_min": -32,
    "lat_max": -20,
    "lon_min": 298,
    "lon_max": 311,
}

#SEAS5 absolute vals
vars=['t2m', 'tp']
SON_ready=SON_mean.rename({'tprate':'tp'})[vars]

SON_Andes_region=select_region(SON_ready, lat_min=-55, lat_max=-35, lon_max=290, lon_min=284)
SON_LP_region=select_region(SON_ready, lat_min=-32, lat_max=-20, lon_min=298, lon_max=311)

Andes_SON_abs=spatial_average(SON_Andes_region).compute()
LP_SON_abs=spatial_average(SON_LP_region).compute()

DJF_ready=DJF_mean.rename({'tprate':'tp'})[vars]
DJF_Andes_region=select_region(DJF_ready, lat_min=-55, lat_max=-35, lon_max=290, lon_min=284)
DJF_LP_region=select_region(DJF_ready,lat_min=-32, lat_max=-20, lon_min=298, lon_max=311 )

Andes_DJF_abs=spatial_average(DJF_Andes_region).compute()
LP_DJF_abs=spatial_average(DJF_LP_region).compute()

print(Andes_DJF_abs)
print('SEAS5 data prepped')

#ERA5 absolute vals

def extract_seasonal_data(array, seasons):
    """
    Extract data from the array for specific seasons (months).
    - A subset of the original array with only the data for the specified months.
    """
    # Make sure the 'time' dimension has a 'month' coordinate
    if 'time' in array.coords:
        # Extract month from the 'time' dimension
        months = array['time'].dt.month
        
        # Filter based on the provided seasons (months)
        seasonal_data = array.sel(time=months.isin(seasons))
        
        return seasonal_data
    else:
        raise ValueError("The input array does not contain a 'time' dimension.")

def shift_december(da):
    time_df = pd.to_datetime(da['time'])
    # Create a boolean mask for times in December
    time_series=pd.Series(time_df)
    mask = time_series.dt.month == 12
    time_series.loc[mask] = time_series.loc[mask] + pd.DateOffset(years=1)
    # Convert to numpy datetime64[D]
    time_df_upd = time_series.values.astype('datetime64[D]')
    #display(time_df_upd)
    d={'time':time_df_upd}
    da_new=da.assign_coords(d)
    return da_new

def seasonal_average(da):
    da2=da.groupby('time.year').mean('time', skipna=True)
    return(da2)


era5_t2m=xr.open_dataset('../data/era5_t2m.nc').rename({'valid_time':'time'})
era5_t2m=era5_t2m.sel(time=slice('1950','2024'))
era5_tp=xr.open_dataset('../data/era5_tp.nc').rename({'valid_time':'time'})
era5_tp=era5_tp.sel(time=slice('1950','2024'))

era5_data=xr.merge([era5_t2m,era5_tp])
print(era5_data)

era5_data_time_avg_SON=seasonal_average(extract_seasonal_data(era5_data, [9,10,11]))
era5_data_time_avg_DJF=seasonal_average(shift_december(extract_seasonal_data(era5_data, [12,1,2])))

SON_Andes_region_era=select_region(era5_data_time_avg_SON,  lat_min=-55, lat_max=-35, lon_max=-70, lon_min=-76)
SON_LP_region_era=select_region(era5_data_time_avg_SON, lat_min=-32, lat_max=-20, lon_min=-62, lon_max=-49)

Andes_SON_abs_era=spatial_average(SON_Andes_region_era) #.rename({'time':'year'})
LP_SON_abs_era=spatial_average(SON_LP_region_era)  #.rename({'time':'year'})

DJF_Andes_region_era=select_region(era5_data_time_avg_DJF, lat_min=-55, lat_max=-35, lon_max=-70, lon_min=-76)
DJF_LP_region_era=select_region(era5_data_time_avg_DJF, lat_min=-32, lat_max=-20, lon_min=-62, lon_max=-49 )

Andes_DJF_abs_era=spatial_average(DJF_Andes_region_era)
LP_DJF_abs_era=spatial_average(DJF_LP_region_era)

print(Andes_DJF_abs_era)
print(np.isnan(Andes_DJF_abs_era).sum())
print('Data prep of absolute values completed!')

#Start plotting
def generate_samples(ds, var, len_boot=None, seed=None):

    rng = np.random.default_rng(seed)

    da = ds[var]

    # --------------------------------------------------
    # Get forecast year for every sample
    # --------------------------------------------------

    years = da.forecast_reference_time.dt.year.values
    unique_years = np.unique(years)

    # Number of years to include in each bootstrap realization
    if len_boot is None:
        len_boot = len(unique_years)

    # --------------------------------------------------
    # Resample YEARS with replacement
    # --------------------------------------------------

    selected_years = rng.choice(
        unique_years,
        size=len_boot,
        replace=True
    )

    # --------------------------------------------------
    # Keep ALL ensemble members / init months belonging
    # to each selected year
    # --------------------------------------------------

    samples = []

    for year in selected_years:

        year_mask = years == year

        values = da.values[year_mask]

        values = values[np.isfinite(values)]

        samples.extend(values)

    return np.asarray(samples)

def generate_samples_fixed_init_month(
    ds,
    var,
    len_boot=None,
    seed=42
):

    rng = np.random.default_rng(seed)

    da = ds[var]

    months = np.unique(
        da.init_month.values
    )

    samples = []

    for month in months:

        da_month = da.sel(
            init_month=month
        )

        years = (
            da_month
            .forecast_reference_time
            .dt.year
            .values
        )

        unique_years = np.unique(years)

        if len_boot is None:
            n_years = len(unique_years)
        else:
            n_years = len_boot

        # ----------------------------------------------
        # Bootstrap years
        # ----------------------------------------------

        selected_years = rng.choice(
            unique_years,
            size=n_years,
            replace=True
        )

        month_samples = []

        # ----------------------------------------------
        # Retain all ensemble members for each year
        # ----------------------------------------------

        for year in selected_years:

            year_mask = years == year

            values = da_month.values[year_mask]

            values = values[np.isfinite(values)]

            month_samples.extend(values)

        samples.append(
            np.asarray(month_samples)
        )

    return samples

def bootstrap_chain(
    ds,
    iterations,
    var,
    sep_lead=False,
    len_boot=None,
    seed=42
):

    rng = np.random.default_rng(seed)

    if sep_lead == False:

        return [
            generate_samples(
                ds,
                var,
                len_boot=len_boot,
                seed=rng.integers(0, 1_000_000_000)
            )
            for _ in range(iterations)
        ]

    else:

        return [
            generate_samples_fixed_init_month(
                ds,
                var,
                len_boot=len_boot,
                seed=rng.integers(0, 1_000_000_000)
            )
            for _ in range(iterations)
        ]


def model_eval_plot(
    boot_samples,
    era5_array,
    input_title,
    ds_init=None
):

    q = np.linspace(0.01, 0.99, 99)

    if ds_init is None:

        # --------------------------------------------------
        # ERA5 quantiles
        # --------------------------------------------------

        era5_values = np.asarray(era5_array)
        era5_values = era5_values[np.isfinite(era5_values)]

        era5_percentiles = np.quantile(
            era5_values,
            q
        )

        # --------------------------------------------------
        # Quantiles for every bootstrap realization
        # --------------------------------------------------

        boot_percentiles = np.array([
            np.quantile(sample, q)
            for sample in boot_samples
        ])

        # Shape:
        # (n_bootstrap, n_quantiles)

        # --------------------------------------------------
        # Actual model distribution
        # --------------------------------------------------

        model_values = ds_init.values if ds_init is not None else None

        # --------------------------------------------------
        # Bootstrap uncertainty
        # --------------------------------------------------

        boot_lower = np.percentile(
            boot_percentiles,
            2.5,
            axis=0
        )

        boot_upper = np.percentile(
            boot_percentiles,
            97.5,
            axis=0
        )

        boot_mean = np.mean(
            boot_percentiles,
            axis=0
        )

        fig = plt.figure(
            figsize=(6, 6)
        )

        plt.plot(
            era5_percentiles,
            era5_percentiles,
            "k-",
            label="1:1 line"
        )

        plt.scatter(
            era5_percentiles,
            boot_mean,
            color="blue",
            label="Bootstrap mean"
        )

        plt.fill_between(
            era5_percentiles,
            boot_lower,
            boot_upper,
            alpha=0.4,
            label="95% bootstrap CI"
        )

        plt.xlabel(
            "ERA5 (observed) data"
        )

        plt.ylabel(
            "Hindcast model data"
        )

        plt.title(
            input_title
        )

        plt.legend()
        plt.grid(alpha=0.3)

        plt.show()

        return (
            boot_percentiles,
            era5_percentiles,
            fig
        )

    else:

        months_list = list(
            np.unique(
                ds_init.init_month.values
            )
        )

        fig, axes = plt.subplots(
            ncols=len(months_list),
            figsize=(
                len(months_list) * 6,
                6
            )
        )

        axes = np.atleast_1d(axes)

        fsize = 16

        for j, month in enumerate(months_list):

            # ------------------------------------------
            # ERA5
            # ------------------------------------------

            era5_values = np.asarray(
                era5_array
            )

            era5_values = era5_values[
                np.isfinite(era5_values)
            ]

            era5_percentiles = np.quantile(
                era5_values,
                q
            )

            # ------------------------------------------
            # Bootstrap quantiles
            # ------------------------------------------

            boot_percentiles = np.array([
                np.quantile(sample, q)
                for sample in boot_samples[j]
            ])

            # ------------------------------------------
            # Mean + uncertainty
            # ------------------------------------------

            boot_mean = np.mean(
                boot_percentiles,
                axis=0
            )

            boot_lower = np.percentile(
                boot_percentiles,
                2.5,
                axis=0
            )

            boot_upper = np.percentile(
                boot_percentiles,
                97.5,
                axis=0
            )

            axes[j].plot(
                era5_percentiles,
                era5_percentiles,
                "k-",
                label="1:1 line"
            )

            axes[j].scatter(
                era5_percentiles,
                boot_mean,
                color="blue",
                label="Bootstrap mean"
            )

            axes[j].fill_between(
                era5_percentiles,
                boot_lower,
                boot_upper,
                alpha=0.4,
                label="95% bootstrap CI"
            )

            axes[j].set_xlabel(
                "ERA5 (observed) data",
                fontsize=fsize
            )

            axes[j].set_ylabel(
                "Hindcast model data",
                fontsize=fsize
            )

            axes[j].tick_params(
                axis="both",
                which="major",
                labelsize=fsize
            )

            axes[j].set_title(
                f"Init month: {months_list[j]}",
                fontsize=fsize
            )

            axes[j].legend(
                fontsize=fsize
            )

            axes[j].grid(
                alpha=0.5
            )

        fig.suptitle(
            input_title,
            fontsize=fsize + 2
        )

        plt.show()

        return (
            boot_percentiles,
            era5_percentiles,
            fig
        )

def time_series_plot(ds, era_var, era5_array, input_title):
    fsize = 16

    n_months = len(np.unique(ds.init_month.values))

    fig, axes = plt.subplots(
        ncols=n_months + 1,
        figsize=(n_months * 6 + 5, 6),
        sharey=True,
        gridspec_kw={"width_ratios": [1] * n_months + [0.8]}
    )

    axes = axes.flatten()

    # Keep the original ERA5 array
    era5_all = era5_array

    for j, month in enumerate(np.unique(ds.init_month.values)):

        ds_month = ds.sel(init_month=month)

        # --------------------------------------------------
        # Select matching ERA5 years
        # --------------------------------------------------

        mask_min = np.min(
            ds_month.forecast_reference_time.dt.year.values
        )

        mask_max = np.max(
            ds_month.forecast_reference_time.dt.year.values
        )

        era5_month = era5_all.sel(
            year=slice(mask_min, mask_max)
        )

        # --------------------------------------------------
        # Time series
        # --------------------------------------------------

        axes[j].scatter(
            ds_month.forecast_reference_time.dt.year.values,
            ds_month,
            color="grey",
            label="Hindcast model"
        )

        axes[j].plot(
            era5_month.year.values,
            era5_month[era_var],
            color="red",
            label="ERA5"
        )

        # --------------------------------------------------
        # Formatting
        # --------------------------------------------------

        axes[j].tick_params(
            axis="both",
            which="major",
            labelsize=fsize
        )

        axes[j].set_title(
            f"Init month: {int(month)}",
            fontsize=fsize
        )

        axes[j].set_xlabel(
            "Forecast reference time",
            fontsize=fsize
        )

        axes[j].set_ylabel(
            era_var,
            fontsize=fsize
        )

        axes[j].legend(fontsize=fsize)
        axes[j].grid(alpha=0.5)

    # --------------------------------------------------
    # Store data for boxplot
    # --------------------------------------------------

      # All model ensemble members / years
    model_values = ds.values.flatten()
    model_values = model_values[np.isfinite(model_values)]

    # Corresponding ERA5 years
    era5_values = era5_month[era_var].values.flatten()
    era5_values = era5_values[np.isfinite(era5_values)]

    # --------------------------------------------------
    # Boxplot on right-hand axis
    # --------------------------------------------------

    parts = axes[-1].violinplot(
    [model_values, era5_values],
    positions=[1, 2],
    showmeans=True,
    showmedians=True,
    showextrema=True
    )

    # Color the violin bodies
    parts["bodies"][0].set_facecolor("grey")
    parts["bodies"][0].set_alpha(0.7)

    parts["bodies"][1].set_facecolor("red")
    parts["bodies"][1].set_alpha(0.7)

    # Mean
    mean_col='blue'
    parts["cmeans"].set_color(mean_col)
    parts["cmeans"].set_linewidth(3)

    # Median
    median_col='orange'
    parts["cmedians"].set_color(median_col)
    parts["cmedians"].set_linewidth(3)

    #extrema
    parts["cmins"].set_color("black")
    parts["cmaxes"].set_color("black")
    parts["cbars"].set_color("black")

    # Legend handles
    legend_handles = [
        Patch(
        facecolor="grey",
        alpha=0.6,
        label="Model"
        ),
        Patch(
            facecolor="red",
            alpha=0.6,
            label="ERA5"
        ),
        Line2D(
            [0], [0],
            color=mean_col,
            linestyle="-",
            linewidth=2,
            label="Mean"
        ),
        Line2D(
            [0], [0],
            color=median_col,
            linestyle="-",
            linewidth=2,
            label="Median"
        )
    ]

    axes[-1].legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=11,
        framealpha=0.8
    )

    # x-axis labels
    axes[-1].set_xticks([1, 2])
    axes[-1].set_xticklabels(["Model", "ERA5"])

    axes[-1].set_title(
        f"Distribution of {era_var} values",
        fontsize=fsize
    )

    axes[-1].tick_params(
        axis="both",
        which="major",
        labelsize=fsize
    )

    axes[-1].grid(
        axis="y",
        alpha=0.5
    )

    fig.suptitle(
        input_title,
        fontsize=fsize + 2
    )

    plt.show()

    return fig

#####################################################################################
#Results
#####################################################################################

folder='Abs_vals/'

vars=['t2m', 'tp']

for var in vars:

    #1. time series
    ts_Andes_SON=time_series_plot(Andes_SON_abs[var], var, Andes_SON_abs_era, 
                                  f'Time series of absolute {var} values in Andes SON')
    ts_Andes_SON.savefig(folder+f'ts_{var}_Andes_SON_comparison.jpg',dpi=300)

    ts_LP_SON_temp=time_series_plot(LP_SON_abs[var], var, LP_SON_abs_era,
                                     f'Time series of absolute {var} values in La Plata SON')
    ts_LP_SON_temp.savefig(folder+f'ts_{var}_LP_SON_comparison.jpg')

    #b) DJF
    ts_Andes_DJF=time_series_plot(Andes_DJF_abs[var], var, Andes_DJF_abs_era, 
                                      f'Time series of absolute {var} values in Andes DJF')
    ts_Andes_DJF.savefig(folder+f'ts_{var}_Andes_DJF_comparison.jpg',dpi=300)
    
    ts_LP_DJF_temp=time_series_plot(LP_DJF_abs[var], var, LP_DJF_abs_era,
                                         f'Time series of absolute {var} values in La Plata DJF')
    ts_LP_DJF_temp.savefig(folder+f'ts_{var}_LP_DJF_comparison.jpg')
    

    print(f'Time series for {var} finished!')
    #2. model eval plots
    boot_Andes_SON_abs=bootstrap_chain(Andes_SON_abs, 200, var=var, sep_lead=True)
    eval_Andes_SON=model_eval_plot(boot_Andes_SON_abs, Andes_SON_abs_era[var], f'Model evaluation of absolute {var} values in Andes SON',
                                   Andes_SON_abs)

    eval_Andes_SON[-1].savefig(folder+f'eval_plot_{var}_Andes_SON_comparison.jpg', dpi=300)

    boot_LP_SON_abs=bootstrap_chain(LP_SON_abs, 200, var=var, sep_lead=True)
    eval_LP_SON=model_eval_plot(boot_LP_SON_abs, LP_SON_abs_era[var], f'Model evaluation of absolute {var} values in La Plata SON',
                                       LP_SON_abs)
    
    eval_LP_SON[-1].savefig(folder+f'eval_plot_{var}_LP_SON_comparison.jpg', dpi=300)

    print(f'Model eval for {var} finished!')
    
print('Script finished!')



