import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

DATADIR = "/climca/data/SEAS5_SA/data"

ds = xr.open_dataset(
    DATADIR + "/seas5_daily_u_component_wind_50hPa_SepInit.nc",
    decode_cf=False,
    chunks={
        "number": 1,
        "forecast_reference_time": 10,
        "forecast_period": 31,
    }
)

from xarray.coding.times import decode_cf_datetime

# Decode forecast_reference_time
forecast_reference_time_decoded = decode_cf_datetime(
    ds["forecast_reference_time"].values,
    units=ds["forecast_reference_time"].attrs["units"],
    calendar=ds["forecast_reference_time"].attrs.get("calendar", "standard"),
)

u = ds["u"].assign_coords(
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

def prepare_u60S(u):
    u_zonal = u.mean(dim="longitude")

    u_60S = u_zonal.interp(latitude=-60)

    if "pressure_level" in u_60S.dims:
        u_60S = u_60S.squeeze("pressure_level", drop=True)

    return u_60S

def smooth_forecast_period(u_60S, window=5):
    return u_60S.rolling(forecast_period=window, center=True).mean()

def _find_breakdown_doy_1d(u_values, valid_times, threshold=10.0, crossing="last"):
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
    return float(date.dayofyear)

def compute_vortex_breakdown_doy(u, threshold=10.0, window=5, crossing="last"):
    u_60S = prepare_u60S(u)
    u_60S_roll = smooth_forecast_period(u_60S, window=window)

    u_60S_roll = u_60S_roll.chunk({"forecast_period": -1})
    valid_time = u_60S_roll["valid_time"].chunk({"forecast_period": -1})

    breakdown_doy = xr.apply_ufunc(
        _find_breakdown_doy_1d,
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

    return breakdown_doy.rename("vortex_breakdown_doy")

breakdown_doy = compute_vortex_breakdown_doy(
    u,
    threshold=10.0,
    window=5,
    crossing="last",
).compute()

print(breakdown_doy)

# Convert to a DataFrame
df = breakdown_doy.to_dataframe(name="VB_DOY").reset_index()

# Save to CSV
df.to_csv("/climca/people/glattus/Hindcast_data_ready/SEAS5_VortexBreakdown_DOY_Sep.csv", index=False)