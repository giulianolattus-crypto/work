import numpy as np
import pandas as pd
import xarray as xr


DATADIR = "/climca/data/SEAS5_SA/data"

ds = xr.open_dataset(
    DATADIR + "/seas5_daily_u_component_wind_50hPa_AugInit.nc"
    , decode_cf=False
    ,
    chunks={
        "number": 1,
        "forecast_reference_time": 10,
        "forecast_period": -1,
    }
)

u = ds

# keep October-Jan forecast window:
# 1440 h = 60 days
# 3600 h = 150 days
#u = u.sel(
#    forecast_period=slice(
#        np.timedelta64(1440, "h"),
#        np.timedelta64(3600, "h"),
#    )
#)

import numpy as np
import pandas as pd
import xarray as xr

DATADIR = "/climca/data/SEAS5_SA/data"

ds = xr.open_dataset(
    DATADIR + "/seas5_daily_u_component_wind_50hPa_AugInit.nc",
    decode_cf=False,
    chunks={
        "number": 1,
        "forecast_reference_time": 10,
        "forecast_period": -1,
    }
)

from xarray.coding.times import decode_cf_datetime

# Decode forecast_reference_time
forecast_reference_time_decoded = decode_cf_datetime(
    ds["forecast_reference_time"].values,
    units=ds["forecast_reference_time"].attrs["units"],
    calendar=ds["forecast_reference_time"].attrs.get("calendar", "standard"),
)

u = ds['u'].assign_coords(
    forecast_reference_time=(
        ds["forecast_reference_time"].dims,
        forecast_reference_time_decoded
    )
)

# Decode valid_time
valid_time_decoded = xr.coding.times.decode_cf_datetime(
    ds["valid_time"].values,
    units=ds["valid_time"].attrs["units"],
    calendar=ds["valid_time"].attrs.get("calendar", "standard"),
)

u = u.assign_coords(
    valid_time=(
        ds["valid_time"].dims,
        valid_time_decoded
    )
)

# ---------------------------------------------------------
# Keep October-January forecast window
# forecast_period is already numeric hours
# 1440 h = 60 days
# 3600 h = 150 days
# ---------------------------------------------------------
u = u.sel(
    forecast_period=slice(
        1440.0,
        3600.0
    )
)

def prepare_u60S(u):
    u_zonal = u.mean(dim="longitude")
    u_60S = u_zonal.interp(latitude=-60)

    if "pressure_level" in u_60S.dims:
        u_60S = u_60S.squeeze("pressure_level", drop=True)

    return u_60S

def smooth_forecast_period(u_60S, window=5):
    return u_60S.rolling(forecast_period=window, center=True).mean()


def _find_breakdown_wrapped_doy_1d(
    u_values,
    valid_times,
    threshold=10.0,
    crossing="last",
):
    ok = np.isfinite(u_values)

    if ok.sum() == 0:
        return np.nan

    below = (u_values < threshold) & ok

    previous_below = np.roll(below, 1)
    previous_below[0] = False

    crossed = below & (~previous_below)

    idx = np.where(crossed)[0]

    if len(idx) == 0:
        return np.nan

    if crossing == "first":
        i = idx[0]
    elif crossing == "last":
        i = idx[-1]
    else:
        raise ValueError("crossing must be 'first' or 'last'")

    date = pd.Timestamp(valid_times[i])
    doy = date.dayofyear

    # for Oct-Jan season: Jan should appear after Dec
    if date.month == 1:
        doy = doy + 365

    return float(doy)

def compute_vortex_breakdown_doy_aug(
    u,
    threshold=10.0,
    window=5,
    crossing="last",
):
    u_60S = prepare_u60S(u)
    u_60S_roll = smooth_forecast_period(u_60S, window=window)

    u_60S_roll = u_60S_roll.chunk({"forecast_period": -1})
    valid_time = u_60S_roll["valid_time"].chunk({"forecast_period": -1})

    breakdown_doy = xr.apply_ufunc(
        _find_breakdown_wrapped_doy_1d,
        u_60S_roll,
        valid_time,
        kwargs={
            "threshold": threshold,
            "crossing": crossing,
        },
        input_core_dims=[["forecast_period"], ["forecast_period"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    return breakdown_doy.rename("VB_DOY")


breakdown_doy_aug = compute_vortex_breakdown_doy_aug(
    u,
    threshold=10.0,
    window=5,
    crossing="last",
).compute()

df = breakdown_doy_aug.to_dataframe(name="VB_DOY").reset_index()

df["latitude"] = -60

df.to_csv(
    "/climca/people/glattus/Hindcast_data_ready/SEAS5_VortexBreakdown_DOY_Aug.csv",
    index=False,
)

print(df.head())