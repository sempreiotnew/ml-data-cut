# features.py
import numpy as np
import pandas as pd
from typing import Any, Tuple, List

def safe_to_float_array(x: Any) -> np.ndarray:
    if x is None:
        return np.array([], dtype=float)
    if isinstance(x, (list, tuple, pd.Series, np.ndarray)):
        s = pd.Series(x)
        return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    if isinstance(x, str):
        sep = ';' if ';' in x else (',' if ',' in x else None)
        if sep:
            parts = [p.strip() for p in x.split(sep) if p.strip() != ""]
            if not parts:
                return np.array([], dtype=float)
            return pd.to_numeric(pd.Series(parts), errors="coerce").to_numpy(dtype=float)
        try:
            return np.array([float(x)], dtype=float)
        except Exception:
            return np.array([np.nan], dtype=float)
    try:
        return np.array([float(x)], dtype=float)
    except Exception:
        return np.array([np.nan], dtype=float)


def _count_runs(mask: np.ndarray) -> int:
    """Count contiguous True runs in boolean mask."""
    if mask.size == 0:
        return 0
    m = mask.astype(int)
    starts = np.where((m == 1) & (np.concatenate(([0], m[:-1])) == 0))[0]
    return int(len(starts))


def _count_recovery_peaks(gas: np.ndarray, baseline: float, threshold: float) -> int:
    """
    Count peaks up as recoveries after drops below (baseline - threshold)
    until gas rises back above that level.
    """
    total = 0
    in_drop = False
    for val in gas:
        if np.isnan(val):
            continue
        if val <= baseline - threshold:
            in_drop = True
        elif in_drop and val > baseline - threshold:
            total += 1
            in_drop = False
    return total

def _compute_drop_time(gas, baseline, threshold_abs, millis):
    # -----------------------
    # Compute total drop time
    # -----------------------
    drop_total_time_s = 0
    recovery_total_time_s = 0

    in_drop = False
    drop_start = 0
    recovery_start = 0
    for i, val in enumerate(gas):
        if np.isnan(val):
            continue
        if val <= baseline - threshold_abs:
            if not in_drop:
                drop_start = millis[i]
                in_drop = True
        else:
            if in_drop:
                drop_end = millis[i]
                drop_total_time_s += (drop_end - drop_start) / 1000.0
                recovery_start = drop_end
                in_drop = False
                # Recovery duration until gas rises above baseline - threshold
                j = i
                while j < len(gas) and gas[j] <= baseline - threshold_abs:
                    j += 1
                if j < len(gas):
                    recovery_end = millis[j]
                    recovery_total_time_s += (recovery_end - recovery_start) / 1000.0    


def _compute_peaks(gas, millis, threshold_pct):
    total_peaks_down = 0
    total_peaks_up = 0

    if gas.size >= 1 and not np.all(np.isnan(gas)) and millis.size == gas.size:
            baseline = gas[0] if not np.isnan(gas[0]) else np.nan
            if np.isnan(baseline) or baseline == 0:
                valid_prefix = gas[~np.isnan(gas)]
                baseline = float(np.nanmean(valid_prefix[:min(5, valid_prefix.size)])) if valid_prefix.size > 0 else np.nan

            if not np.isnan(baseline):
                threshold_abs = abs(baseline) * threshold_pct
                valid = ~np.isnan(gas)
                drop_mask = (gas <= baseline - threshold_abs) & valid
                total_peaks_down = _count_runs(drop_mask)
                total_peaks_up = _count_recovery_peaks(gas, baseline, threshold_abs)

                
    return total_peaks_down, total_peaks_up


# # -------------------------------
# # Drop Area
# # -------------------------------
# def compute_drop_area(gas_arr: Any, millis_arr: Any, baseline: float = None) -> float:
#     gas = safe_to_float_array(gas_arr)
#     millis = safe_to_float_array(millis_arr) / 1000.0  # convert to seconds
#     if gas.size == 0 or millis.size == 0:
#         return np.nan
#     baseline = baseline if baseline is not None else gas[0]
#     min_idx = np.nanargmin(gas)
#     drop_area = np.trapezoid(baseline - gas[:len(gas)], x=millis[:len(millis)])
#     return float(drop_area)


# # -------------------------------
# # Recovery Area
# # -------------------------------
# def compute_recovery_area(gas_arr: Any, millis_arr: Any) -> float:
#     gas = safe_to_float_array(gas_arr)
#     millis = safe_to_float_array(millis_arr) / 1000.0  # seconds
#     if gas.size == 0 or millis.size == 0:
#         return np.nan
#     min_idx = np.nanargmin(gas)
#     recovery_area = np.trapezoid(y=gas[0:], x=millis[0:])
#     return float(recovery_area)


# # -------------------------------
# # Slope Drop
# # -------------------------------
# def compute_slope_drop(gas_arr: Any, millis_arr: Any) -> float:
#     gas = safe_to_float_array(gas_arr)
#     millis = safe_to_float_array(millis_arr) / 1000.0  # seconds
#     if gas.size < 2 or millis.size < 2:
#         return np.nan
#     # segment from start to min value
#     min_idx = np.nanargmin(gas)
#     slope_drop = (gas[min_idx] - gas[0]) / (millis[min_idx] - millis[0])
#     return float(slope_drop)


# # -------------------------------
# # Slope Recovery
# # -------------------------------
# def compute_slope_recovery(gas_arr: Any, millis_arr: Any) -> float:
#     gas = safe_to_float_array(gas_arr)
#     millis = safe_to_float_array(millis_arr) / 1000.0  # seconds
#     if gas.size < 2 or millis.size < 2:
#         return np.nan
#     # segment from min value to end
#     min_idx = np.nanargmin(gas)
#     slope_recovery = (gas[-1] - gas[min_idx]) / (millis[-1] - millis[min_idx])
#     return float(slope_recovery)

import numpy as np
from typing import Any

def safe_to_float_array(arr: Any) -> np.ndarray:
    return np.array(arr, dtype=float)

import numpy as np
from typing import Any

def safe_to_float_array(arr: Any) -> np.ndarray:
    return np.array(arr, dtype=np.float64)

# -------------------------------
# Slope Drop
# -------------------------------
def compute_slope_drop(gas_arr: Any, millis_arr: Any) -> float:
    gas = safe_to_float_array(gas_arr)
    millis = safe_to_float_array(millis_arr) / 1000.0  # convert to seconds
    if gas.size < 2 or millis.size < 2:
        return 0.0

    max_idx = np.nanargmax(gas)
    # Find the first minimum **after** this max
    if max_idx >= gas.size - 1:
        return 0.0  # no drop possible
    min_idx = gas[max_idx:].argmin() + max_idx

    delta_gas = gas[max_idx] - gas[min_idx]
    delta_time = millis[min_idx] - millis[max_idx]

    # Only positive drop
    if delta_gas <= 0 or delta_time <= 0:
        return 0.0

    return float(delta_gas / delta_time)


# -------------------------------
# Slope Recovery
# -------------------------------
def compute_slope_recovery(gas_arr: Any, millis_arr: Any) -> float:
    gas = safe_to_float_array(gas_arr)
    millis = safe_to_float_array(millis_arr) / 1000.0
    if gas.size < 2 or millis.size < 2:
        return 0.0

    min_idx = np.nanargmin(gas)
    # Recovery is max **after** min
    if min_idx >= gas.size - 1:
        return 0.0  # no recovery
    max_after_min_idx = gas[min_idx:].argmax() + min_idx

    delta_gas = gas[max_after_min_idx] - gas[min_idx]
    delta_time = millis[max_after_min_idx] - millis[min_idx]

    # Only positive recovery
    if delta_gas <= 0 or delta_time <= 0:
        return 0.0

    return float(delta_gas / delta_time)

def extract_features(
    sensor_id,    
    gas_arr: Any,
    temp_arr: Any = None,
    pressure_arr: Any = None,
    humidity_arr: Any = None,
    millis_arr: Any = None,
    threshold_pct: float = 0.20,
) -> Tuple[np.ndarray, List[str]]:

    gas = safe_to_float_array(gas_arr)
    temp = safe_to_float_array(temp_arr)
    press = safe_to_float_array(pressure_arr)
    hum = safe_to_float_array(humidity_arr)
    millis = safe_to_float_array(millis_arr)  # in milliseconds

    def _stats(a: np.ndarray):
        if a.size == 0 or np.all(np.isnan(a)):
            return (np.nan, np.nan, np.nan, np.nan)
        return (
            float(np.nanmin(a)),
            float(np.nanmax(a)),
            float(np.nanmean(a)),
            float(np.nanstd(a, ddof=0)),
        )

    gas_min, gas_max, gas_mean, gas_std = _stats(gas)
    t_min, t_max, t_mean, t_std = _stats(temp)
    p_min, p_max, p_mean, p_std = _stats(press)
    h_min, h_max, h_mean, h_std = _stats(hum)

    total_peaks_down, total_peaks_up =_compute_peaks(gas=gas, millis=millis, threshold_pct=threshold_pct)
    # recovery_area_oms_seconds = compute_recovery_area(gas_arr=gas, millis_arr=millis)
    # drop_area_oms_seconds = compute_drop_area(gas_arr=gas, millis_arr=millis, baseline=gas_mean)
    slope_recovery = compute_slope_recovery(gas_arr=gas, millis_arr=millis)
    slope_drop = compute_slope_drop(gas_arr=gas, millis_arr=millis)
    feature_names = [
        "sensor_id",
        "gas_min", "gas_max", "gas_mean", "gas_std",
        "temperature_min", "temperature_max", "temperature_mean", "temperature_std",
        "pressure_min", "pressure_max", "pressure_mean", "pressure_std",
        "humidity_min", "humidity_max", "humidity_mean", "humidity_std",
        "total_peaks_up", "total_peaks_down",
        "slope_drop",
        "slope_recovery"
    ]
    
    vals = [
        sensor_id,
        gas_min, gas_max, gas_mean, gas_std,
        t_min, t_max, t_mean, t_std,
        p_min, p_max, p_mean, p_std,
        h_min, h_max, h_mean, h_std,
        float(total_peaks_up), float(total_peaks_down),
        slope_drop,
        slope_recovery
    ]
    

    return np.array(vals, dtype=float), feature_names

