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

spring_leads_list=[]
for init_month in [6,7,8,9]:
    ##spring season block
    lead_months_spring=season_to_leads(init_month=init_month, season_months=[9,10,11])
    print(lead_months_spring)
    block_spring=make_3_month_block(ds, lead_months=lead_months_spring, init_month=init_month)
    spring_leads_list.append(block_spring)

summer_leads_list=[]
for init_month in [9,10,11,12]:
    ##summer season block
    lead_months_summer=season_to_leads(init_month=init_month, season_months=[12,1,2])
    print(lead_months_summer)
    block_summer=make_3_month_block(ds, lead_months=lead_months_summer, init_month=init_month)
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

def compute_sample_anomaly(ds):
    """
    Compute anomaly across all samples.

    For each grid point:
        anomaly = value - mean over samples

    This keeps all ensemble members and all years.
    """

    return ds - ds.mean(dim="sample")

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


# Andes temperature, SON and DJF
t_Andes_ssavg_SON = process_region(SON_mean[["t2m"]], **ANDES)
t_Andes_ssavg_DJF = process_region(DJF_mean[["t2m"]], **ANDES)

# La Plata temperature, SON and DJF
t_LaPlata_ssavg_SON = process_region(SON_mean[["t2m"]], **LA_PLATA)
t_LaPlata_ssavg_DJF = process_region(DJF_mean[["t2m"]], **LA_PLATA)


# Andes precipitation, SON and DJF
pr_era_Andes_ssavg_SON = process_region(SON_mean[["tprate"]], **ANDES)
pr_era_Andes_ssavg_DJF = process_region(DJF_mean[["tprate"]], **ANDES)

# La Plata precipitation, SON and DJF
pr_era_LaPlata_ssavg_SON = process_region(SON_mean[["tprate"]], **LA_PLATA)
pr_era_LaPlata_ssavg_DJF = process_region(DJF_mean[["tprate"]], **LA_PLATA)

def rename_precip(ds):
    if 'tprate' in ds.data_vars:
        ds = ds.rename({"tprate": "tp"})
    return ds

pr_era_Andes_ssavg_SON=rename_precip(pr_era_Andes_ssavg_SON)
pr_era_Andes_ssavg_DJF=rename_precip(pr_era_Andes_ssavg_DJF)
pr_era_LaPlata_ssavg_DJF=rename_precip(pr_era_LaPlata_ssavg_DJF)
pr_era_LaPlata_ssavg_SON=rename_precip(pr_era_LaPlata_ssavg_SON)

save_path='/climca/people/glattus/'
save_folder='Hindcast_data_ready'


##Temp and precip united by region
ds_Andes_SON=xr.merge([t_Andes_ssavg_SON, pr_era_Andes_ssavg_SON])
ds_Andes_DJF=xr.merge([t_Andes_ssavg_DJF, pr_era_Andes_ssavg_DJF])
ds_LP_SON=xr.merge([t_LaPlata_ssavg_SON, pr_era_LaPlata_ssavg_SON])
ds_LP_DJF=xr.merge([t_LaPlata_ssavg_DJF, pr_era_LaPlata_ssavg_DJF])

#reset index
ds_Andes_SON=ds_Andes_SON.reset_index('sample')
ds_Andes_DJF=ds_Andes_DJF.reset_index('sample')
ds_LP_SON=ds_LP_SON.reset_index('sample')
ds_LP_DJF=ds_LP_DJF.reset_index('sample')
#print(ds_Andes_SON)
encoding = {var: {"zlib": True, "complevel": 4} for var in ds_Andes_SON.data_vars}
ds_Andes_DJF.to_netcdf(save_path+save_folder+'/Target_vars_Andes_DJF.nc', encoding=encoding)
print('Summer Andes saved')
ds_Andes_SON.to_netcdf(save_path+save_folder+'/Target_vars_Andes_SON.nc', encoding=encoding)
print('Spring Andes saved')
ds_LP_DJF.to_netcdf(save_path+save_folder+'/Target_vars_LP_DJF.nc', encoding=encoding)
print('LP Summer saved')
ds_LP_SON.to_netcdf(save_path+save_folder+'/Target_vars_LP_SON.nc', encoding=encoding)
print('All saved!')


print('Temperature and Precipitation Computation finished!')