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
        sample=('season', "forecast_reference_time", "number")
    )
    ds_3m_DJF = ds_3m_DJF.stack(
        sample=('season', "forecast_reference_time", "number")
    )

    

    ds_3m_SON = ds_3m_SON.transpose(
        "sample",
        'init_month',
        "season_month",
        "latitude",
        "longitude"
    )

    ds_3m_DJF = ds_3m_DJF.transpose(
        "sample",
        'init_month',
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


def process_init_month_sep(ds, func, **kwargs):
    results = []

    for init_month in ds.init_month:
        out = func(ds.sel(init_month=init_month), **kwargs)
        out = out.expand_dims(init_month=[init_month.item()])
        results.append(out)

    ds_out=xr.concat(results, dim="init_month")
    ds_out=ds_out.chunk({'sample':1000})
    return ds_out

##Functions for ENSO & IOD
def ENSO_process(sst):
    'Computation of Nino3.4 index from sst field (ideally already monthly)'

    print('Start ENSO comp')
    sst_nino = select_region(
        sst,
        lat_min=-5,
        lat_max=5,
        lon_min=190,
        lon_max=240
    ) #fixed region
    sst_anomaly_nino=compute_sample_anomaly(sst_nino)
    sst_anomaly_nino_weighted=spatial_average(sst_anomaly_nino)
    
    nino_detr=detrend_across_time(sst_anomaly_nino_weighted)
    print('Detrended ENSO')
    
    sst_anomaly_smoothed = nino_detr.rolling(sample=5, center=True).mean()
    nino34_index=standardize(sst_anomaly_smoothed)
    nino34_index=nino34_index.rename({'sst':'nino34'})
    print('ENSO index computed')
    return nino34_index


from scipy.signal import butter, filtfilt

# Function to apply a low-pass filter (removes >7-year variations)

def lowpass_filter(da, cutoff=1/7, fs=12):
    nyq=fs/2
    wn=cutoff/nyq #standardising my input frequency with nyquist
    # Design the Butterworth filter
    b, a = butter(N=6, Wn=wn, btype='high', fs=fs) 
    #actually it is high pass because low frequency signal acts on long timescales

    # Define a function to apply the filter on 1D time series
    def filter_1d(x):
        if np.any(np.isnan(x)):  # Handle NaNs
            x = np.nan_to_num(x, nan=np.nanmean(x))  # Replace NaNs with mean
        return filtfilt(b, a, x)

    # Apply the filter along the 'time' dimension
    filtered_da = xr.apply_ufunc(
        filter_1d, 
        da, 
        input_core_dims=[["sample"]],  # Apply only along 'sample'
        output_core_dims=[["sample"]],
        vectorize=True  # Ensures it works for each lat/lon point
    )

    return filtered_da

# Function to apply a 3-month running mean
def running_mean(da, window=3):
    return da.rolling(time=window, center=True).mean()

def IOD_process(sst):
    ##modify to adapt worklfow of julia
    '''Computation procedure for IOD index'''
    print('Start IOD comp')
    sst_io_eastweighted=select_region(sst, lat_min=-10, 
                                      lat_max=0,
                                      lon_min=90,
                                      lon_max=110)
    sst_io_westweighted=select_region(sst, lat_min=-10,
                                      lat_max=10,
                                      lon_max=70,
                                      lon_min=50)
   
    #weight by lat and spatial avg after anomaly comp
    sst_anom_ioe=spatial_average(compute_sample_anomaly(sst_io_eastweighted))
    sst_anom_iow=spatial_average(compute_sample_anomaly(sst_io_westweighted))
    print('Anomaly computed')
    #detrend
    ioe_detr=detrend_across_time(sst_anom_ioe).compute()
    iow_detr=detrend_across_time(sst_anom_iow).compute()

    #filter high freqs
    print('Start lowpass')
    ioe_filtered=lowpass_filter(ioe_detr)
    iow_filtered=lowpass_filter(iow_detr)
    #smooth and compute IOD index
    iode_smoothed=ioe_filtered.rolling(sample=3, center=True).mean()
    iodw_smoothed=iow_filtered.rolling(sample=3, center=True).mean()
    iod_index=iodw_smoothed-iode_smoothed
    iod_index_ssavg=standardize(iod_index).rename({'sst':'iod_index'})
    print('IOD index computed')
    return iod_index_ssavg

################################################################
#Start ENSO and IOD comp

sst_SON=SON_mean[['sst']]
sst_DJF=DJF_mean[['sst']]
nino_SON=process_init_month_sep(sst_SON, ENSO_process)
nino_DJF=process_init_month_sep(sst_DJF, ENSO_process)
IOD_SON=process_init_month_sep(sst_SON, IOD_process)


## save ocean drivers in nc file
save_path='/climca/people/glattus/'
save_folder='Hindcast_data_ready'

#save SON indices
ds_ocean_SON=xr.merge([IOD_SON, nino_SON])
ds_ocean_SON=ds_ocean_SON.reset_index('sample')
encoding_ocean = {var: {"zlib": True, "complevel": 4} for var in ds_ocean_SON.data_vars}
ds_ocean_SON.to_netcdf(save_path+save_folder+'/ENSO_IOD_SON_init_sep.nc', encoding=encoding_ocean)
print('Spring data saved!')



#####################################################################
#Additional functions for summer indian ocean mode comp
#####################################################################

def plot_eof1(Vt, X_2d, title="EOF1 pattern"):
    eof1 = Vt[0, :]

    eof1_map = xr.DataArray(
        eof1,
        coords={"space": X_2d.space},
        dims=["space"],
        name="EOF1"
    ).unstack("space")

    eof1_map.plot()
    plt.title(title)
    plt.savefig('Index_comp/EOF1_IOBW')
    plt.close()
    return eof1_map

def plot_pc1(pc1_ts, title="PC1 time series"):
    pc1_ts.plot(x='forecast_reference_time',
                          marker='.', linestyle='none', label='IOB DJF')
    plt.axhline(0, color='black', linestyle='dashed')
    plt.legend()
    plt.title(title)
    plt.savefig('Index_comp/PC1_IOBW.jpg')
    plt.close()


def IOBW_process(sst):

    # 1. Select Indian Ocean region
    sst_io_whole = select_region(
        sst,
        lat_max=26,
        lat_min=-26,
        lon_max=120,
        lon_min=30
    ).sortby("latitude", ascending=True)

    # 2. Anomaly
    sst_anom = compute_sample_anomaly(sst_io_whole)
    print("Anomaly computed")

    # 3. Detrend
    sst_detr = detrend_across_time(sst_anom)

    # 4. Latitude weighting (IMPORTANT: apply on grid, not sample)
    sst_weighted = weight_by_latitude(sst_detr)
    print('Weighted by latitude')
    X = sst_weighted["sst"]

    X_2d = X.stack(space=("latitude", "longitude"))
    X_2d = X_2d.transpose("sample", "space")
    X_2d = X_2d.fillna(np.nanmean(X_2d))


    print('Safety checks!')
    print("Shape:", X_2d.shape)
    print("Size (GB):", X_2d.nbytes / 1024**3)  
    print("Finite:", np.isfinite(X_2d).all())
    print("NaNs:", np.isnan(X_2d).sum())
    print("Infs:", np.isinf(X_2d).sum())
    
    
    print("Performing PCA...")

    # 7. SVD / PCA
    U, S, Vt = perform_svd(X_2d.values)

    print(U.shape, S.shape, Vt.shape)
    eof1_map = plot_eof1(Vt, X_2d)

    # 8. PC1 time series
    PC1 = U[:, 0] * S[0]

    PC1_ts = xr.DataArray(
        PC1,
        coords={"sample": X_2d.sample},
        dims=["sample"],
        name="IOBW"
    )

    # 9. Sign convention
    PC1_ts = -PC1_ts

    # 10. Standardization
    IOBW_index = standardize(PC1_ts)
    plot_pc1(IOBW_index)
    
    print("PCA finished")

    return IOBW_index


IOBW_index=process_init_month_sep(sst_DJF, IOBW_process)
print('IOBW computation finished!')

for i, element in enumerate([nino_SON, nino_DJF, IOD_SON, IOBW_index]):
    fig=plt.figure(figsize=(10, 4))

    for init in np.unique(element.init_month.values):
        if isinstance(element,xr.Dataset):
            var=list(element.data_vars)[0]
        elif isinstance(element,xr.DataArray):
            var=element.name

        element.sel(init_month=init)[var].plot(
                        x="forecast_reference_time",
                        marker=".",
                        linestyle="none",
                        label=f" {var} {init}",alpha=0.5
                    )


    plt.axhline(0, color="black", linestyle="dashed")
    plt.legend()
    

    plt.title(f"{var}")
    fig.savefig(f'Index_comp/{var}_ts.jpg', dpi=300)

#save the DJF indices
ds_ocean_DJF=xr.merge([IOBW_index, nino_DJF])
ds_ocean_DJF=ds_ocean_DJF.reset_index('sample')
encoding_ocean_DJF = {var: {"zlib": True, "complevel": 4} for var in ds_ocean_DJF.data_vars}
ds_ocean_DJF.to_netcdf(save_path+save_folder+'/ENSO_IOB_DJF_init_sep.nc', encoding=encoding_ocean_DJF)

print('Summer data saved!')