import pandas as pd
from cycles_signal_process import (
    prepare_wave_data,
    group_pqrst_points,
    calculate_amplitudes,
    calculate_intervals,
    calculate_amplitude_differences,
    calculate_all_wave_areas,
    calculate_area_ratios,

    calc_average_signal,
    calculate_statistics,
)



def calc_signal_morphology_features(cleaned_signal, fs, waves_peak_info, avg_signal=False):
    features = pd.Series(dtype=float)
    waves_peak_info = prepare_wave_data(waves_peak_info)
    grouped_cycles = group_pqrst_points(waves_peak_info)
    grouped_cycles = calculate_amplitudes(grouped_cycles, cleaned_signal)
    grouped_cycles = calculate_intervals(grouped_cycles, fs)
    grouped_cycles = calculate_amplitude_differences(grouped_cycles)

    grouped_cycles = calculate_all_wave_areas(grouped_cycles, cleaned_signal, fs)
    grouped_cycles = calculate_area_ratios(grouped_cycles)
    stats = calculate_statistics(grouped_cycles, avg_signal=avg_signal)
    features = pd.Series(stats)
    
    return features