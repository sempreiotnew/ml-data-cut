# features.py
import numpy as np
import pandas as pd
from typing import Any, Optional, Tuple, List, Dict

def safe_to_float_array(x: Any) -> np.ndarray:
    """Convert input to a float numpy array. Keep NaNs if conversion fails."""
    if x is None:
        return np.array([], dtype=float)
    if isinstance(x, np.ndarray):
        return x.astype(float, copy=False)
    if isinstance(x, (list, tuple, pd.Series)):
        return np.array([float(i) if pd.notna(i) else np.nan for i in x], dtype=float)
    if isinstance(x, str):
        sep = ';' if ';' in x else (',' if ',' in x else None)
        if sep:
            parts = [p.strip() for p in x.split(sep) if p.strip()]
            return np.array([float(p) if p.replace('.', '', 1).isdigit() else np.nan for p in parts], dtype=float)
        try:
            return np.array([float(x)], dtype=float)
        except:
            return np.array([np.nan], dtype=float)
    try:
        return np.array(x, dtype=float)
    except:
        return np.array([np.nan], dtype=float)


def compute_fingerprint(t: np.ndarray, y: np.ndarray, rel_threshold: float = 0.05) -> Dict[str, Any]:
    """Compute dynamic features of the gas response using proper time axis (t in seconds)."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    n = len(y)
    out: Dict[str, Any] = {}
    out['n_points'] = n
    if n == 0:
        return out

    # Baseline
    baseline_n = max(1, int(0.05 * n))
    baseline_vals = y[:baseline_n]
    baseline_mean = float(np.nanmean(baseline_vals))
    baseline_median = float(np.nanmedian(baseline_vals))
    baseline_std = float(np.nanstd(baseline_vals))
    out.update({
        'baseline_mean': baseline_mean,
        'baseline_median': baseline_median,
        'baseline_std': baseline_std
    })

    # Drop
    min_idx = int(np.nanargmin(y))
    min_val = float(y[min_idx])
    drop_magnitude = baseline_mean - min_val
    drop_start_idx_candidates = np.where((baseline_mean - y) / baseline_mean >= rel_threshold)[0]
    drop_start_idx = int(drop_start_idx_candidates[0]) if drop_start_idx_candidates.size > 0 else 0
    drop_start_time_s = float(t[drop_start_idx])
    out.update({
        'min_idx': min_idx,
        'min_val': min_val,
        'drop_magnitude': drop_magnitude,
        'drop_start_idx': drop_start_idx,
        'drop_start_time_s': drop_start_time_s
    })

    # Recovery
    recovery_candidates = np.where(y >= baseline_mean)[0]
    recovery_candidates = recovery_candidates[recovery_candidates > min_idx]
    recovery_end_idx = int(recovery_candidates[0]) if recovery_candidates.size > 0 else n - 1

    drop_duration_s = float(t[min_idx] - t[drop_start_idx])
    recovery_duration_s = float(t[recovery_end_idx] - t[min_idx])
    total_response_time_s = float(t[recovery_end_idx] - t[drop_start_idx])
    out.update({
        'recovery_end_idx': recovery_end_idx,
        'drop_duration_s': drop_duration_s,
        'recovery_duration_s': recovery_duration_s,
        'total_response_time_s': total_response_time_s
    })

    # Rates (velocity of drop and recovery)
    out['drop_rate'] = drop_magnitude / drop_duration_s if drop_duration_s > 0 else 0.0
    out['recovery_rate'] = (y[recovery_end_idx] - min_val) / recovery_duration_s if recovery_duration_s > 0 else 0.0

    # Slopes
    dy = np.diff(y)
    dt = np.diff(t)
    slopes = dy / np.where(dt == 0, np.nan, dt)
    out['max_negative_slope'] = float(np.nanmin(slopes)) if slopes.size else 0.0
    out['max_positive_slope'] = float(np.nanmax(slopes)) if slopes.size else 0.0
    out['mean_slope'] = float(np.nanmean(slopes)) if slopes.size else 0.0

    # Areas
    if min_idx > drop_start_idx:
        drop_area = float(np.trapezoid(baseline_mean - y[drop_start_idx:min_idx+1], x=t[drop_start_idx:min_idx+1]))
    else:
        drop_area = 0.0

    if recovery_end_idx > min_idx:
        recovery_area = float(np.trapezoid(y[min_idx:recovery_end_idx+1] - min_val, x=t[min_idx:recovery_end_idx+1]))
    else:
        recovery_area = 0.0

    out['drop_area'] = drop_area
    out['recovery_area'] = recovery_area
    out['area_symmetry'] = drop_area / recovery_area if recovery_area > 0 else 0.0

    # Misc
    out['skewness'] = float(pd.Series(y).skew())
    out['kurtosis'] = float(pd.Series(y).kurtosis())
    out['inflection_points'] = int(np.sum(np.abs(np.diff(np.sign(np.diff(y)))) > 0)) if len(y) > 2 else 0
    out['final_value'] = float(y[-1])
    out['recovery_percentage'] = (y[-1] - min_val) / drop_magnitude * 100 if drop_magnitude > 0 else 0.0
    out['time_to_min_fraction'] = (t[min_idx] - t[0]) / (t[-1] - t[0]) if t[-1] > t[0] else 0.0
    out['peaks_before_min'] = sum((y[i] > y[i-1] and y[i] > y[i+1]) for i in range(1, min_idx)) if min_idx > 1 else 0
    out['peaks_after_min'] = sum((y[i] > y[i-1] and y[i] > y[i+1]) for i in range(min_idx+1, n-1)) if n - min_idx > 2 else 0
    
    return out


def extract_features(gas_arr: Any,
                     temp_arr: Optional[Any] = None,
                     pressure_arr: Optional[Any] = None,
                     humidity_arr: Optional[Any] = None,
                     sample_interval_s: Optional[float] = None,
                     millis_arr: Optional[Any] = None,
                     rel_threshold: float = 0.05
                     ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:

    gas = safe_to_float_array(gas_arr)
    temp = safe_to_float_array(temp_arr)
    press = safe_to_float_array(pressure_arr)
    hum = safe_to_float_array(humidity_arr)
    millis = safe_to_float_array(millis_arr).astype(float) if millis_arr is not None else None

    # Build time axis (t in seconds)
    if millis is not None and len(millis) == len(gas):
        t = (millis - millis[0]) / 1000.0
    else:
        t = np.arange(len(gas), dtype=float) * (sample_interval_s if sample_interval_s else 1.0)

    fingerprint = compute_fingerprint(t, gas, rel_threshold=rel_threshold)

    def _stats(a: np.ndarray):
        a = np.asarray(a, dtype=float)
        if a.size == 0 or np.all(np.isnan(a)):
            return (np.nan, np.nan, np.nan, np.nan)
        return (float(np.nanmean(a)), float(np.nanmin(a)), float(np.nanmax(a)), float(np.nanstd(a)))

    # Stats for all signals
    gas_mean, gas_min, gas_max, gas_std = _stats(gas)
    t_mean, t_min, t_max, t_std = _stats(temp)
    p_mean, p_min, p_max, p_std = _stats(press)
    h_mean, h_min, h_max, h_std = _stats(hum)

    feature_names = [
        'n_points', 'baseline_mean', 'baseline_median', 'baseline_std',
        'min_val', 'min_idx', 'drop_magnitude', 'drop_start_idx', 'drop_start_time_s',
        'drop_duration_s', 'recovery_end_idx', 'recovery_duration_s', 'total_response_time_s',
        'drop_rate', 'recovery_rate', 'max_negative_slope', 'max_positive_slope',
        'mean_slope', 'drop_area', 'recovery_area', 'area_symmetry',
        'skewness', 'kurtosis', 'inflection_points', 'peaks_before_min',
        'peaks_after_min', 'recovery_percentage', 'final_value', 'time_to_min_fraction',
        'gas_mean', 'gas_min', 'gas_max', 'gas_std',
        'temperature_mean', 'temperature_min', 'temperature_max', 'temperature_std',
        'pressure_mean', 'pressure_min', 'pressure_max', 'pressure_std',
        'humidity_mean', 'humidity_min', 'humidity_max', 'humidity_std'
    ]

    vals = [
        fingerprint.get('n_points', 0.0), fingerprint.get('baseline_mean', 0.0),
        fingerprint.get('baseline_median', 0.0), fingerprint.get('baseline_std', 0.0),
        fingerprint.get('min_val', 0.0), fingerprint.get('min_idx', 0.0),
        fingerprint.get('drop_magnitude', 0.0), fingerprint.get('drop_start_idx', 0.0),
        fingerprint.get('drop_start_time_s', 0.0), fingerprint.get('drop_duration_s', 0.0),
        fingerprint.get('recovery_end_idx', 0.0), fingerprint.get('recovery_duration_s', 0.0),
        fingerprint.get('total_response_time_s', 0.0), fingerprint.get('drop_rate', 0.0),
        fingerprint.get('recovery_rate', 0.0), fingerprint.get('max_negative_slope', 0.0),
        fingerprint.get('max_positive_slope', 0.0), fingerprint.get('mean_slope', 0.0),
        fingerprint.get('drop_area', 0.0), fingerprint.get('recovery_area', 0.0),
        fingerprint.get('area_symmetry', 0.0), fingerprint.get('skewness', 0.0),
        fingerprint.get('kurtosis', 0.0), fingerprint.get('inflection_points', 0.0),
        fingerprint.get('peaks_before_min', 0.0), fingerprint.get('peaks_after_min', 0.0),
        fingerprint.get('recovery_percentage', 0.0), fingerprint.get('final_value', 0.0),
        fingerprint.get('time_to_min_fraction', 0.0),
        gas_mean, gas_min, gas_max, gas_std,
        t_mean, t_min, t_max, t_std,
        p_mean, p_min, p_max, p_std,
        h_mean, h_min, h_max, h_std
    ]

    return np.array(vals), feature_names, fingerprint
