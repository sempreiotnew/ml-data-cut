# features.py
import numpy as np
import pandas as pd
from typing import Any, Tuple, List
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

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


def safe_to_float_array(arr: Any) -> np.ndarray:
    return np.array(arr, dtype=float)

def compute_standard_deviation(gas_arr : np.ndarray):
    gas = np.asarray(gas_arr, dtype=float)
    n = gas.size
    
    if n < 2:
        return np.array([], dtype=float)
    
    stds = np.empty(n - 1, dtype=float)
    
    for i in range(1, n):
        stds[i-1] = np.nanstd([gas[i-1], gas[i]])
    
    return stds





def calculate_area_fixed_segments(gas_arr: np.ndarray, segment_size: int = 50, fixed_baseline: float = 1e5):
    plt.figure(figsize=(12,6))
    plt.plot(gas_arr, label="Gas Sensor", color="gray")

    drop_areas = []

    # Split the signal into fixed segments
    for start in range(0, len(gas_arr), segment_size):
        end = min(start + segment_size, len(gas_arr))
        x_region = np.arange(start, end)
        y_region = gas_arr[start:end]

        # Area = curve down to fixed baseline
        drop_values = y_region - fixed_baseline
        total_area = np.trapezoid(drop_values)  # integrate
        drop_areas.append(total_area)

        # Fill area under curve to baseline
        plt.fill_between(x_region, fixed_baseline, y_region, color="orange", alpha=0.3)

        # Annotate area
        mid_x = x_region[len(x_region)//2]
        mid_y = fixed_baseline + np.max(drop_values)/2
        plt.text(mid_x, mid_y, f"{total_area:.0f}", ha='center', va='bottom', fontsize=8, color='black')

    plt.axhline(fixed_baseline, color="green", linestyle="--", label=f"Baseline ({fixed_baseline})")
    plt.xlabel("Sample Index")
    plt.ylabel("Gas Resistance")
    plt.title(f"Gas Sensor Signal with Drop Areas ({segment_size}-measurement segments)")
    plt.legend()
    plt.show()

    return drop_areas

def find_peaks_segment(gas_arr: np.ndarray, fixed_baseline: float = 1e5):
    data_min = np.min(gas_arr)
    data_max = np.max(gas_arr)
    rangePeak = 0.1 * (data_max - data_min)
    rangeValley = 0.02 * (data_max - data_min)

    # Find peaks
    peaks, _ = find_peaks(
        gas_arr,
        height=data_min + rangePeak,
        prominence=rangePeak,
        distance=5
    )

    # Find valleys
    valleys, _ = find_peaks(
        -gas_arr,
        prominence=rangeValley,
        distance=5
    )

    plt.figure(figsize=(12,6))
    plt.plot(gas_arr, label="Gas Sensor", color="gray")

    plt.plot(peaks, gas_arr[peaks], "x", color="blue", label="Peaks")
    plt.plot(valleys, gas_arr[valleys], "x", color="red", label="Valleys")
    plt.axhline(fixed_baseline, color="green", linestyle="--", label=f"Baseline ({fixed_baseline})")
    plt.xlabel("Sample Index")
    plt.ylabel("Gas Resistance")
    plt.title("Gas Sensor Signal with Drop Areas (Fixed Baseline)")
    plt.legend()
    plt.show()

    return peaks, valleys


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
            # float(np.nanstd(a, ddof=0)),
        )

    gas_min, gas_max, gas_mean = _stats(gas)
    t_min, t_max, t_mean = _stats(temp)
    p_min, p_max, p_mean = _stats(press)
    h_min, h_max, h_mean = _stats(hum)
    gas_std = compute_standard_deviation(gas_arr=gas_arr)
    area = calculate_area_fixed_segments(gas_arr=gas)
    peaks, valleys = find_peaks_segment(gas_arr=gas_arr)

    feature_names = [
        "sensor_id",
        "gas_min", "gas_max", "gas_mean",
        "temperature_min", "temperature_max", "temperature_mean", 
        "pressure_min", "pressure_max", "pressure_mean", 
        "humidity_min", "humidity_max", "humidity_mean", 
        "gas_std",
        "area",
        "peaks",
        "valleys"
    ]
    
    vals = [
        sensor_id,
        gas_min, gas_max, gas_mean,
        t_min, t_max, t_mean, 
        p_min, p_max, p_mean, 
        h_min, h_max, h_mean,
        gas_std,
        area,
        peaks,
        valleys
    ]
    

    return np.array(vals, dtype=object), feature_names

if __name__ == "__main__":
    import pandas as pd
    import sys

    # if len(sys.argv) < 2:
    #     print("Usage: python features.py <csv_file>")
    #     sys.exit(1)

    # csv_file = sys.argv[1]

    # Load CSV
    df = pd.read_csv("teste3.csv")
    
    # Keep everything that is NOT in bad_ids
    df = df[~df["id"].isin([765269850, 765286747, 765279836, 765271387, 765283665, 765268824, 765287253])]

    # Expected columns: gas, temperature, pressure, humidity, millis
    gas = df["gas_resistance"].values if "gas_resistance" in df else []
    temp = df["temperature"].values if "temperature" in df else []
    press = df["pressure"].values if "pressure" in df else []
    hum = df["humidity"].values if "humidity" in df else []
    millis = df["millis"].values if "millis" in df else []

    # for d in df['gas_resistance']:
    #     print(d)

    features, names = extract_features(
            sensor_id=765270362,
            gas_arr=gas,
            temp_arr=temp,
            pressure_arr=press,
            humidity_arr=hum,
            millis_arr=millis,
            threshold_pct=0.20
    )
    
    print("\nExtracted Features:")
    for n, v in zip(names, features):
        if isinstance(v, (list, np.ndarray)):
            v_str = ", ".join(f"{x:.2f}" for x in v)
            print(f"{n:20s}: [{v_str}]")
        else:
            print(f"{n:20s}: {v:.2f}")

