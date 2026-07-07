import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA
import scipy.signal as signal
import statsmodels.api as sm
import pandas as pd
from seaborn import regplot
from seaborn import heatmap



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
    pc_map = reshaped_da.isel(sample=0).copy() #.isel selects the i you give, make a copy of given array at i
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
    zonal_map = xr.zeros_like(da.isel(sample=0)) #blank map with same dimensions as da
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


def plot_regression_coefficients(sample, regression_coefficients):
    """Plot regression coefficients over time."""
    plt.figure(figsize=(8, 5))
    plt.plot(sample, regression_coefficients, label='Regression Coefficients')
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

def drop_fake_members(da):
    valid_sample = (
        da.isel(season_month=0, latitude=slice(0, 5), longitude=slice(0, 5))
        .mean(("latitude", "longitude"), skipna=True)
        .notnull()
    )
    return da.where(valid_sample, drop=True)


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

def standardize(ds):
    """
    Standardize each variable across samples.

    z = value / std

    The mean should already be near zero after anomaly + detrending.
    If you want strict z-score, use:
        (ds - ds.mean("sample")) / ds.std("sample")
    """

    return ds / ds.std(dim="sample")


def compute_sample_anomaly(ds):
    """
    Compute anomaly across all samples.

    For each grid point:
        anomaly = value - mean over samples

    This keeps all ensemble members and all years.
    """

    return ds - ds.mean(dim="sample")


   

#full SAM procedure
def SAM_process_fast(zg):
    """Put in seasonal values of Z500 field."""

    # If input is a Dataset, select the variable z
    if isinstance(zg, xr.Dataset):
        zg = zg["z"]

    field_io_whole = select_region(
        zg,
        lat_max=-20,
        lat_min=-90,
        lon_max=360,
        lon_min=0
    )

    field_anom = compute_sample_anomaly(field_io_whole)
    field_anom_seasonal = field_anom.mean(dim="season_month")
    print("Anomaly computed")

    field_anom_weighted = weight_by_latitude(field_anom_seasonal)

    field_anom_weighted = field_anom_weighted.transpose(
        "sample", "latitude", "longitude"
    ).load()

    reshaped_field_anom = field_anom_weighted.stack(
        space=("latitude", "longitude")
    )

    # Use to_numpy(), and force numeric dtype
    X = reshaped_field_anom.to_numpy()
    X = np.asarray(X, dtype=np.float64)

    valid_space = np.isfinite(X).any(axis=0)

    reshaped_field_anom = reshaped_field_anom.isel(space=valid_space)
    X = X[:, valid_space]

    X = X - np.nanmean(X, axis=0, keepdims=True)
    X = np.nan_to_num(X, nan=0.0)

    print("Performing PCA...")

    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    SAM = plot_first_principal_component(reshaped_field_anom, Vt)

    SAM_zonal_mean = compute_zonal_mean(SAM)
    SAM_sym = make_zonal_mean_map(SAM_zonal_mean, field_anom_seasonal)
    SAM_asym = SAM - SAM_sym

    SAM_sym_coefficient = regress_onto_zonal_mean_map(field_anom_seasonal, SAM_sym)
    # plot_regression_coefficients(field_anom_seasonal["sample"], SAM_sym_coefficient)

    SAM_asym_coefficient = regress_onto_zonal_mean_map(field_anom_seasonal, SAM_asym)
    # plot_regression_coefficients(field_anom_seasonal["sample"], SAM_asym_coefficient)

    A_SAM_da = xr.DataArray(
        SAM_asym_coefficient,
        coords={"sample": field_anom_seasonal["sample"].values},
        dims=["sample"],
        name="A_SAM"
    )

    S_SAM_da = xr.DataArray(
        SAM_sym_coefficient,
        coords={"sample": field_anom_seasonal["sample"].values},
        dims=["sample"],
        name="S_SAM"
    )

    d = {
        "SAM": SAM,
        "SAM_zonal_mean": SAM_zonal_mean,
        "SAM_sym": SAM_sym,
        "SAM_asym": SAM_asym,
        "SAM_sym_coefficient": S_SAM_da,
        "SAM_asym_coefficient": A_SAM_da,
        "U": U,
        "S": S,
        "Vt": Vt,
    }

    return d


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

###########################################################################
#1. Start code
##########################################################################


##load data
DATADIR='/climca/data/SEAS5_SA/data'

z500_data=xr.open_dataset(DATADIR+'/seas5_pressure_levels_06-12month_1-6leadtimemonth.nc', 
                          chunks={'number':1, 'forecast_reference_time':10, 'forecastMonth':6})

z500_data=make_valid_time(z500_data)


##########################################################################################
##helpful functions in data preparation
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

def make_3_month_block(ds, lead_months, init_month):
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

    #combine both seasons
    ds_3m = xr.concat(
        [ds_3m_SON, ds_3m_DJF],
        dim="sample"
    )

    ds_3m = ds_3m.transpose(
        "sample",
        "season_month",
        "latitude",
        "longitude"
    )
    return ds_3m
########################################################################################


z500_data_3m_SON = xr.concat(
    [make_3_month_block(z500_data, season_to_leads(init_month=init_month, season_months=[9,10,11]), init_month=init_month) for init_month in [6,7,8,9]],
    dim="init_month")

z500_data_3m_DJF = xr.concat(
    [make_3_month_block(z500_data, season_to_leads(init_month=init_month, season_months=[12,1,2]), init_month=init_month) for init_month in [9,10,11,12]],
    dim="init_month")

z500_data_3m=create_sample_and_cobmine_seasons(z500_data_3m_SON.sel(pressure_level=500), z500_data_3m_DJF.sel(pressure_level=500))


SON_z500 = z500_data_3m.where(z500_data_3m["season"] == "SON", drop=True)

DJF_z500 = z500_data_3m.where(z500_data_3m["season"] == "DJF", drop=True)




#Start SAM computation
SAM_dict_SON=SAM_process_fast(SON_z500)
SAM_dict_DJF=SAM_process_fast(DJF_z500)

#Plot both indices to check
figure=plt.figure(figsize=(10, 5))
plt.plot(SAM_dict_SON['SAM_sym_coefficient'].values, label='S-SAM')
plt.plot(SAM_dict_SON['SAM_asym_coefficient'].values, label='A-SAM')
plt.legend()
plt.title('Spring SAM')
figure.savefig('SAM_Hindcast_SON.png')

figure_DJF=plt.figure(figsize=(10, 5))
plt.plot(SAM_dict_DJF['SAM_sym_coefficient'].values, label='S-SAM')
plt.plot(SAM_dict_DJF['SAM_asym_coefficient'].values, label='A-SAM')
plt.legend()
plt.title('Summer SAM')
figure_DJF.savefig('SAM_Hindcast_DJF.png')

#convert into xarray
SAM_ds_SON=xr.merge([SAM_dict_SON['SAM_sym_coefficient'], SAM_dict_SON['SAM_asym_coefficient']])

SAM_ds_DJF=xr.merge([SAM_dict_DJF['SAM_sym_coefficient'], SAM_dict_DJF['SAM_asym_coefficient']])

#save xarray

encoding = {var: {"zlib": True, "complevel": 4} for var in SAM_ds_SON.data_vars}
encoding_DJF = {var: {"zlib": True, "complevel": 4} for var in SAM_ds_DJF.data_vars}

SAM_ds_SON.to_netcdf('/climca/people/glattus/Hindcast_data_ready/SAM_ready_SON.nc', encoding=encoding)
SAM_ds_DJF.to_netcdf('/climca/people/glattus/Hindcast_data_ready/SAM_ready_DJF.nc', encoding=encoding_DJF)