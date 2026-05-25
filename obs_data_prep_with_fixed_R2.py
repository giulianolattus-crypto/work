import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
import matplotlib.pyplot as plt


# =========================================================
# UTILITIES
# =========================================================

def haversine_matrix(lats, lons):

    lats = np.radians(lats)
    lons = np.radians(lons)

    lat_diff = np.subtract.outer(lats, lats)
    lon_diff = np.subtract.outer(lons, lons)

    a = (
        np.sin(lat_diff / 2) ** 2
        + np.cos(lats[:, None])
        * np.cos(lats[None, :])
        * np.sin(lon_diff / 2) ** 2
    )

    return 2 * 6371 * np.arcsin(np.sqrt(a))


# =========================================================
# BASE CLASS
# =========================================================

class ClimateVariable:

    def __init__(self, input_path, var_name):

        self.input_path = input_path
        self.var_name = var_name

        self.ds = None
        self.ds_sel = None
        self.ds_anomaly = None
        self.ds_clean = None
        self.ds_filled = None

        self.df_stats = None
        self.summary = None
        self.pareto_summary = None

    # =====================================================
    # LOAD CR2 DATA
    # =====================================================

    def load_cr2(self):

        file = self.input_path

        # find data start
        with open(file, encoding='latin1') as f:

            for i, line in enumerate(f):

                if line[:4].isdigit():
                    data_start = i
                    break

        with open(file, encoding='latin1') as f:
            lines = f.readlines()

        station_ids = lines[0].strip().split(',')[1:]

        lat = None
        lon = None

        for line in lines[:data_start]:

            parts = line.strip().split(',')
            key = parts[0].lower()

            if key.startswith("lat"):
                lat = np.array(parts[1:], dtype=float)

            elif key.startswith("lon"):
                lon = np.array(parts[1:], dtype=float)

        # load observations
        df = pd.read_csv(
            file,
            skiprows=data_start,
            na_values=-9999,
            encoding='latin1'
        )

        df.columns = ['date'] + station_ids

        df['date'] = pd.to_datetime(df['date'])

        da = xr.DataArray(

            df.iloc[:, 1:].values,

            dims=("time", "points"),

            coords={
                "time": df["date"].values,
                "points": station_ids,
                "lat": ("points", lat),
                "lon": ("points", lon),
            },

            name=self.var_name
        )

        self.ds = da.to_dataset()

        print(f"{self.var_name} dataset loaded successfully")

    # =====================================================
    # DATA SELECTION
    # =====================================================

    def select_data(
        self,
        start='1950',
        end='2018',
        lat_min=-90,
        lat_max=-15,
        lon_min=-76,
        lon_max=-49
    ):

        ds_sel = self.ds.sel(
            time=slice(start, end)
        )

        ds_sel = ds_sel.where(
            (ds_sel.lat <= lat_max)
            & (ds_sel.lat >= lat_min)
            & (ds_sel.lon >= lon_min)
            & (ds_sel.lon <= lon_max),
            drop=True
        )

        self.ds_sel = ds_sel

        print("Data selection completed")

    # =====================================================
    # PREPROCESS
    # =====================================================

    def preprocess(self):
        """
        Overwrite in subclasses if needed
        """
        pass

    # =====================================================
    # ANOMALIES
    # =====================================================

    def compute_anomaly(self, min_count=3):

        da = self.ds_sel[self.var_name]

        count = da.groupby('time.month').count('time')

        clim = da.groupby('time.month').mean(
            'time',
            skipna=True
        )

        clim = clim.where(count >= min_count)

        anomaly = da.groupby('time.month') - clim

        self.ds_anomaly = anomaly.to_dataset(
            name=self.var_name
        )

        print("Anomaly computed successfully")

    # =====================================================
    # QUALITY CONTROL
    # =====================================================

    def apply_qc(self):

        data = self.ds_anomaly[self.var_name].values

        df = pd.DataFrame(data)

        corr_matrix = df.corr()

        valid_pairs = np.abs(corr_matrix) > 0.8

        scores = np.zeros_like(data)

        months = self.ds_anomaly['time'].dt.month.values

        n_stations = data.shape[1]

        for i in range(n_stations):

            for j in range(i + 1, n_stations):

                if not valid_pairs.values[i, j]:
                    continue

                diff = data[:, i] - data[:, j]

                for m in range(1, 13):

                    mask = months == m

                    idx = np.where(mask)[0]

                    if len(idx) < 10:
                        continue

                    diff_m = diff[idx]

                    valid = ~np.isnan(diff_m)

                    diff_m = diff_m[valid]

                    idx_valid = idx[valid]

                    if len(diff_m) < 10:
                        continue

                    std_m = np.std(diff_m)

                    if std_m == 0 or np.isnan(std_m):
                        continue

                    norm_diff = diff_m / std_m

                    p5, p95 = np.percentile(
                        norm_diff,
                        [5, 95]
                    )

                    outliers = (
                        (norm_diff < p5)
                        | (norm_diff > p95)
                    )

                    scores[idx_valid[outliers], i] += 1
                    scores[idx_valid[outliers], j] += 1

        total_score_per_time = scores.sum(
            axis=1,
            keepdims=True
        )

        fractional_score = np.divide(
            scores,
            total_score_per_time,
            where=total_score_per_time != 0
        )

        mask_bad = fractional_score > 0.1

        clean_data = data.copy()

        clean_data[mask_bad] = np.nan

        clean_ds = xr.DataArray(
            clean_data,
            coords=self.ds_anomaly[self.var_name].coords,
            dims=self.ds_anomaly[self.var_name].dims
        )

        self.ds_clean = clean_ds.to_dataset(
            name=self.var_name
        )

        print("Quality control applied successfully")

    # =====================================================
    # GAPFILL USING YOUR ORIGINAL FUNCTION
    # =====================================================

    def gapfill_station(
    self,
    min_cal_points=30,
    r2_threshold=None,
    pval_threshold=0.05
    ):

        ds = self.ds_clean.copy()

        months = ds['time'].dt.month.values

        n_stations = ds.sizes['points']

        prev_day = ds[self.var_name].shift(time=1)

        next_day = ds[self.var_name].shift(time=-1)

        filled_total = 0

        for target in range(n_stations):

            print(f"Filling station {target}")

            for m in range(1, 13):

                mask = months == m

                y = (
                    ds[self.var_name]
                    .isel(points=target)
                    .values[mask]
                    .astype(float)
                )

                if np.all(~np.isnan(y)):
                    continue

                # predictors
                X_list = []

                x_prev = (
                    prev_day
                    .isel(points=target)
                    .values[mask]
                    .astype(float)
                )

                x_next = (
                    next_day
                    .isel(points=target)
                    .values[mask]
                    .astype(float)
                )

                X_list.append(x_prev)
                X_list.append(x_next)

                for j in range(n_stations):

                    if j == target:
                        continue

                    x_station = (
                        ds[self.var_name]
                        .isel(points=j)
                        .values[mask]
                        .astype(float)
                    )

                    if np.sum(~np.isnan(x_station)) >= min_cal_points:
                        X_list.append(x_station)

                if len(X_list) == 0:
                    continue

                X = np.column_stack(X_list)

                valid_cal = ~np.isnan(y)

                if np.sum(valid_cal) < min_cal_points:
                    continue

                y_cal = y[valid_cal]

                X_cal = X[valid_cal, :]

                X_clean = []

                good_cols = []

                for k in range(X_cal.shape[1]):

                    if not np.isnan(X_cal[:, k]).any():

                        X_clean.append(X_cal[:, k])

                        good_cols.append(k)

                if len(X_clean) == 0:
                    continue

                X_clean = np.column_stack(X_clean)

                # STEPWISE SELECTION
                selected = []

                remaining = list(range(X_clean.shape[1]))

                while remaining:

                    best_p = np.inf
                    best_var = None

                    for var in remaining:

                        try:

                            model = sm.OLS(
                                y_cal,
                                sm.add_constant(
                                    X_clean[:, selected + [var]]
                                )
                            ).fit()

                            worst_p = np.max(model.pvalues[1:])

                            if worst_p < best_p:
                                best_p = worst_p
                                best_var = var

                        except Exception:
                            continue

                    if (
                        best_var is not None
                        and best_p < pval_threshold
                    ):

                        selected.append(best_var)

                        remaining.remove(best_var)

                    else:
                        break

                if len(selected) == 0:
                    continue

                final_model = sm.OLS(
                    y_cal,
                    sm.add_constant(X_clean[:, selected])
                ).fit()

                r2 = final_model.rsquared

                # APPLY FIXED THRESHOLD
                if r2 < r2_threshold:
                    continue

                missing_idx = np.where(np.isnan(y))[0]

                time_index = np.where(mask)[0]

                for idx in missing_idx:

                    try:

                        x_pred_full = np.array(
                            [x[idx] for x in X_list]
                        )

                        x_pred_filtered = (
                            x_pred_full[good_cols]
                        )

                        x_pred = (
                            x_pred_filtered[selected]
                        )

                        if np.isnan(x_pred).all():
                            continue

                        x_pred = np.concatenate(
                            ([1.0], x_pred)
                        ).reshape(1, -1)

                        if (
                            x_pred.shape[1]
                            != len(final_model.params)
                        ):
                            continue

                        y_pred = (
                            final_model.predict(x_pred)[0]
                        )

                        ds[self.var_name].values[
                            time_index[idx],
                            target
                        ] = y_pred

                        filled_total += 1

                    except Exception:
                        continue

        self.ds_filled = ds

        print(f"\nTotal values filled: {filled_total}")

        return ds
    


# =========================================================
# PRECIPITATION CLASS
# =========================================================

class PrecipitationData(ClimateVariable):

    def preprocess(self):

        self.ds[self.var_name] = (
            self.ds[self.var_name]
            .clip(min=0)
        )

        print("Precipitation preprocessing completed")


# =========================================================
# TEMPERATURE CLASS
# =========================================================

class TemperatureData(ClimateVariable):

    def preprocess(self):

        self.ds[self.var_name] = (
            self.ds[self.var_name] - 273.15
        )

        print("Temperature preprocessing completed")


# =========================================================
# MAIN
# =========================================================

print('Program started')

precip = PrecipitationData(

    input_path='/climca/data/RAW_OBS_DATA/CR2_monthly/cr2_prAmon_2018_ghcn/cr2_prAmon_2018_ghcn.txt',

    var_name='precip'
)

temperature=TemperatureData(

    input_path='/climca/data/RAW_OBS_DATA/CR2_monthly/cr2_tasAmon_2018_ghcn/cr2_tasAmon_2018_ghcn.txt',

    var_name='temperature'
)

print('Input data loaded')

# workflow
# PRECIPITATION
precip.load_cr2()
precip.preprocess()
precip.select_data()
precip.compute_anomaly()
precip.apply_qc()

precip.gapfill_station(
    r2_threshold=0.9
)

print("Precipitation processing completed successfully")


# TEMPERATURE
temperature.load_cr2()
temperature.preprocess()
temperature.select_data()
temperature.compute_anomaly()
temperature.apply_qc()

temperature.gapfill_station(
    r2_threshold=0.8
)

print("Temperature processing completed successfully")