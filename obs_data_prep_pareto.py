import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
import matplotlib.pyplot as plt

class ClimateVariable:

    def __init__(self, input_path, var_name):
        self.input_path = input_path
        self.var_name = var_name

        self.ds = None

    def load_cr2(self):

        with open(self.input_path, encoding='latin1') as f:
            for i, line in enumerate(f):
                if line[:4].isdigit():
                    data_start = i
                    break

        with open(self.input_path, encoding='latin1') as f:
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

        df = pd.read_csv(
            self.input_path,
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

        print(f"{self.var_name} dataset loaded")


    ##functions
    #compute anomaly with safe handling of NaNs
    def compute_anomaly(da, min_count=3):
        """
        Compute monthly anomalies, handling NaNs robustly.
        
        Parameters
        ----------
        da : xarray.DataArray
            Input data (time, ...)
        min_count : int
            Minimum number of valid values required to compute climatology
            
        Returns
        -------
        anomaly : xarray.DataArray
        """
        da = self.ds[self.var_name]

        # count valid values per month
        count = da.groupby('time.month').count('time')
        
        # compute climatology (ignore NaNs explicitly)
        clim = da.groupby('time.month').mean('time', skipna=True)
        
        # mask climatology where not enough data
        clim = clim.where(count >= min_count)
        print(clim['precip'].shape)
        # compute anomaly
        anomaly = da.groupby('time.month') - clim

        self.ds_anomaly = anomaly.to_dataset(
            name=self.var_name
        )

        # debug info
        print("NaNs in climatology:", clim['precip'].isnull().sum().values)
        print("Months with insufficient data:", (count['precip'] < min_count).sum().values)
        
        return anomaly

    #quality control function from paper
    def QC(data):
        # convert to DataFrame (columns = stations)
        df = pd.DataFrame(data)

        # compute correlation (automatically ignores NaNs pairwise)
        corr_matrix = df.corr()
        # find valid pairs with condition from paper
        valid_pairs = np.abs(corr_matrix) > 0.8
        print(np.sum(valid_pairs.values))
        ##compute absolute scores
        scores = np.zeros_like(data)
        months = ds_sel['time'].dt.month.values  # important!

        n_stations = data.shape[1]

        for i in range(n_stations):
            for j in range(i+1, n_stations):
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
                    p5, p95 = np.percentile(norm_diff, [5, 95])
                    outliers = (norm_diff < p5) | (norm_diff > p95)

                    scores[idx_valid[outliers], i] += 1
                    scores[idx_valid[outliers], j] += 1
        # sum over stations per time
        total_score_per_time = scores.sum(axis=1, keepdims=True)

        # avoid division by zero
        fractional_score = np.divide(
            scores,
            total_score_per_time,
            where=total_score_per_time != 0
        )
        mask_bad = fractional_score > 0.1

        # apply mask
        clean_data = data.copy()
        clean_data[mask_bad] = np.nan
        clean_ds = xr.DataArray(
            clean_data,
            coords=ds_sel.coords,
            dims=ds_sel.dims
        )
        print("Fractional max:", np.nanmax(fractional_score))
        print("Number removed:", np.sum(mask_bad))
        
        return clean_ds

    ##distance matrix using haversine formula
    def haversine_matrix(lats, lons):
        lats = np.radians(lats)
        lons = np.radians(lons)

        lat_diff = np.subtract.outer(lats, lats)
        lon_diff = np.subtract.outer(lons, lons)

        a = (np.sin(lat_diff / 2) ** 2 +
            np.cos(lats[:, None]) * np.cos(lats[None, :]) *
            np.sin(lon_diff / 2) ** 2)

        return 2 * 6371 * np.arcsin(np.sqrt(a))

    #function for filling gaps with lowest R2 possible and get statistics out of it for pareto condition
    def gapfill_station_all_stats(
        ds,
        min_cal_points=30,
        r2_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5,
                    0.6, 0.7, 0.8, 0.9, 1.0),
        pval_threshold=0.05
    ):

        ds_filled = ds.copy()
        months = ds['time'].dt.month.values
        n_stations = ds.dims['points']

        prev_day = ds['precip'].shift(time=1)
        next_day = ds['precip'].shift(time=-1)

        dist = haversine_matrix(ds.lat.values, ds.lon.values)

        stats_records = []

        for target in range(n_stations):
            print(f"Filling station {target}")

            for m in range(1, 13):
                mask = months == m

                y = ds['precip'].isel(points=target).values[mask].astype(float)

                if np.all(~np.isnan(y)):
                    continue

                # ---------------- predictors ----------------
                X_list = []

                x_prev = prev_day.isel(points=target).values[mask].astype(float)
                x_next = next_day.isel(points=target).values[mask].astype(float)

                X_list.append(x_prev)
                X_list.append(x_next)

                for j in range(n_stations):
                    if j == target:
                        continue

                    x_station = ds['precip'].isel(points=j).values[mask].astype(float)

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

                # remove NaN predictors
                X_clean = []
                good_cols = []

                for k in range(X_cal.shape[1]):
                    if not np.isnan(X_cal[:, k]).any():
                        X_clean.append(X_cal[:, k])
                        good_cols.append(k)

                if len(X_clean) == 0:
                    continue

                X_clean = np.column_stack(X_clean)

                # ---------------- STEPWISE ----------------
                selected = []
                remaining = list(range(X_clean.shape[1]))

                while remaining:
                    best_p = np.inf
                    best_var = None

                    for var in remaining:
                        try:
                            model = sm.OLS(
                                y_cal,
                                sm.add_constant(X_clean[:, selected + [var]])
                            ).fit()

                            worst_p = np.max(model.pvalues[1:])

                            if worst_p < best_p:
                                best_p = worst_p
                                best_var = var

                        except Exception:
                            continue

                    if best_var is not None and best_p < pval_threshold:
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

                #get R2 and RMSE error from final model
                r2 = final_model.rsquared
                rmse = np.sqrt(final_model.mse_resid)

                #print(f"Station {target}, Month {m}, R²: {r2:.3f}")

                missing_idx = np.where(np.isnan(y))[0]
                time_index = np.where(mask)[0]

                # ---------------- precompute valid prediction mask ----------------
                valid_pred_mask = []

                for idx in missing_idx:
                    try:
                        x_pred_full = np.array([x[idx] for x in X_list])
                        x_pred_filtered = x_pred_full[good_cols]
                        x_pred = x_pred_filtered[selected]

                        if np.isnan(x_pred).all():
                            valid_pred_mask.append(False)
                        else:
                            valid_pred_mask.append(True)

                    except Exception:
                        valid_pred_mask.append(False)

                valid_pred_mask = np.array(valid_pred_mask)
                valid_pred_indices = missing_idx[valid_pred_mask]

                # ---------------- thresholds ----------------
                for thr in r2_thresholds:

                    filled_count = 0

                    if r2 < thr:
                        stats_records.append({
                            "station": target,
                            "month": m,
                            "r2": r2,
                            'rmse':rmse,
                            "threshold": thr,
                            "filled": 0,
                            "possible": len(valid_pred_indices)
                        })
                        continue

                    for idx in valid_pred_indices:
                        try:
                            x_pred_full = np.array([x[idx] for x in X_list])
                            x_pred_filtered = x_pred_full[good_cols]
                            x_pred = x_pred_filtered[selected]

                            x_pred = np.concatenate(([1.0], x_pred)).reshape(1, -1)

                            if x_pred.shape[1] != len(final_model.params):
                                continue

                            y_pred = final_model.predict(x_pred)[0]

                            # fill only once (lowest threshold)
                            if thr == min(r2_thresholds):
                                ds_filled['precip'].values[target, time_index[idx]] = y_pred

                            filled_count += 1

                        except Exception:
                            continue

                    stats_records.append({
                        "station": target,
                        "month": m,
                        "r2": r2,
                        'rmse':rmse,
                        "threshold": thr,
                        "filled": filled_count,
                        "possible": len(valid_pred_indices)
                    })

        # ---------------- results ----------------
        df_stats = pd.DataFrame(stats_records)

        summary = (
            df_stats
            .groupby("threshold")[["filled", "possible"]]
            .sum()
        )

        summary["fraction_filled"] = summary["filled"] / summary["possible"]

        print("\n=== OVERVIEW ===")
        print(summary)

        # sanity check (should be non-positive)
        print("\nMonotonic check (should be <= 0):")
        print(summary["filled"].diff())

        return ds_filled, df_stats, summary

    #pareto scoring using RMSE

    def pareto_scoring_rmse(
        df,
        alpha=0.5,
        ):
        """
        Pareto optimization using:
            - coverage  (higher is better)
            - RMSE      (lower is better)

        alpha : float
            Weight for coverage.
            0.5 = equal balance

        rmse_col : str
            Name of RMSE column
        """

        # ---------------- AGGREGATE ----------------
        summary = (
            df.groupby("threshold")
            .apply(
                lambda g: pd.Series({

                    # fraction of gaps filled
                    "coverage":
                        g["filled"].sum() / g["possible"].sum(),

                    # weighted average RMSE
                    "mean_rmse":
                        np.average(
                            g['rmse'],
                            weights=g["filled"].clip(lower=1)
                        )
                })
            )
            .reset_index()
        )

        # ---------------- PLOT TRADEOFF ----------------
        fig, ax1 = plt.subplots(figsize=(7,5))

        ax1.plot(
            summary["threshold"],
            summary["coverage"],
            marker='o'
        )

        ax1.set_ylabel("Coverage")
        ax1.set_xlabel("R² threshold")

        ax2 = ax1.twinx()

        ax2.plot(
            summary["threshold"],
            summary["mean_rmse"],
            color='red',
            marker='o'
        )

        ax2.set_ylabel("Mean RMSE")

        plt.title("Coverage vs RMSE Tradeoff")
        plt.show()

        # ---------------- NORMALIZATION ----------------
        coverage = summary["coverage"].values
        rmse = summary["mean_rmse"].values

        # lower RMSE is better
        rmse_norm = (
            (rmse.max() - rmse)
            / (rmse.max() - rmse.min())
        )

        # ---------------- PARETO SCORE ----------------
        score = (
            alpha * coverage
            + (1 - alpha) * rmse_norm
        )

        summary["score"] = score

        # ---------------- BEST THRESHOLD ----------------
        best = summary.iloc[np.argmax(score)]

        print("\n===== BEST THRESHOLD =====")
        print(best)

        # ---------------- SCORE PLOT ----------------
        plt.figure(figsize=(6,5))

        plt.plot(
            summary["mean_rmse"],
            summary["score"],
            marker='o'
        )

        plt.xlabel("Mean RMSE")
        plt.ylabel("Pareto Score")
        plt.title("Pareto Score vs RMSE")

        plt.gca().invert_xaxis()

        plt.savefig("pareto_score.png", dpi=300)

        # ---------------- PARETO FRONT ----------------
        plt.figure(figsize=(6,5))

        plt.plot(
            summary["coverage"],
            summary["mean_rmse"],
            marker='o'
        )

        for _, row in summary.iterrows():
            plt.text(
                row["coverage"],
                row["mean_rmse"],
                f"{row['threshold']:.1f}"
            )

        plt.xlabel("Coverage")
        plt.ylabel("Mean RMSE")

        plt.title("Pareto Tradeoff")

        plt.gca().invert_yaxis()

        plt.savefig("pareto_tradeoff.png", dpi=300)

        return summary

##read data


file = '/climca/data/RAW_OBS_DATA/CR2_monthly/cr2_prAmon_2018_ghcn/cr2_prAmon_2018_ghcn.txt'

# --- FIND DATA START ---
with open(file, encoding='latin1') as f:
    for i, line in enumerate(f):
        if line[:4].isdigit():   # line starts with year
            data_start = i
            break

# --- READ HEADER LINES ---
with open(file, encoding='latin1') as f:
    lines = f.readlines()

# existing: station IDs
header = lines[0].strip().split(',')
station_ids = header[1:]

# 🔥 NEW: extract lat, lon, basin codes
lat = None
lon = None
basin = None

for line in lines[:data_start]:
    parts = line.strip().split(',')
    key = parts[0].lower()

    if key.startswith("lat"):
        lat = np.array(parts[1:], dtype=float)

    elif key.startswith("lon"):
        lon = np.array(parts[1:], dtype=float)

# --- LOAD DATA ---
df = pd.read_csv(
    file,
    skiprows=data_start,
    na_values=-9999,
    encoding='latin1'
)

df.columns = ['date'] + station_ids
df['date'] = pd.to_datetime(df['date'])

# --- TO XARRAY ---
da = xr.DataArray(
    df.iloc[:, 1:].values,
    dims=("time", "points"),
    coords={
        "time": df["date"].values,
        "points": station_ids,
        "lat": ("points", lat),
        "lon": ("points", lon),
    },
    name="precip"
)

ds = da.to_dataset()
print('Dataset opened successfully')

## data selection and anomaly computation
##1 select timeframe and area frame
ds_sel=ds.sel(time=slice('1950','2018'))
ds_sel = ds_sel.where(
    (ds_sel.lat <= -15) & (ds_sel.lat >= -90) &
    (ds_sel.lon >= -76) & (ds_sel.lon <= -49),
    drop=True
)
ds_monthly = ds_sel #.resample(time="1MS").sum()
##2 compute anomaly
ds_an=compute_anomaly(ds_monthly)

data_an=ds_an['precip'].values

print('Anomaly computed successfully')

##quality control
ds_clean=QC(data_an).to_dataset(name='precip')
print('Quality control applied successfully')

##gapfilling and its statistsics using pareto condition

ds_filled, df_stats, summary = gapfill_station_all_stats(ds_clean)
print('Gap filling statistics computed successfully')

##find best threshold using pareto scoring
pareto_summary = pareto_scoring_rmse(df_stats)
print('Pareto scoring completed successfully')