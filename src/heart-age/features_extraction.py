import pandas as pd
import neurokit2 as nk
import numpy as np 
from scipy.signal import savgol_filter
from collections import defaultdict
from astropy.timeseries import LombScargle
import pywt
from scipy import stats
from antropy import sample_entropy

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

def calculate_feature_statistics(feature_values, prefix=""):
    features = pd.Series(dtype=float)
    
    for feat_name, values in feature_values.items():
        if not values:
            continue
            
        stats = {
            f"{prefix}{feat_name}_mean": np.mean(values),
            f"{prefix}{feat_name}_std": np.std(values),
            f"{prefix}{feat_name}_min": np.min(values),
            f"{prefix}{feat_name}_max": np.max(values),
            f"{prefix}{feat_name}_median": np.median(values),
            f"{prefix}{feat_name}_q25": np.quantile(values, 0.25),
            f"{prefix}{feat_name}_q75": np.quantile(values, 0.75),
        }
        
        features = pd.concat([features, pd.Series(stats)])
    
    return features

element_config = [
    # P-волна
    {'name': 'P','points': {'start': 'ECG_P_Onsets','peak': 'ECG_P_Peaks','end': 'ECG_P_Offsets'}},
    # R-зубец
    {'name': 'R','points': {'start': 'ECG_R_Onsets', 'peak': 'ECG_R_Peaks','end': 'ECG_R_Offsets'}},
    # S-комплекс (от R до конца S)
    {'name': 'S','points': {'start': 'ECG_R_Peaks', 'peak': 'ECG_S_Peaks','end': 'ECG_R_Offsets'}},
    # QRS-комплекс
    {'name': 'QRS', 'points': {'start': 'ECG_Q_Peaks','peak': 'ECG_R_Peaks','end': 'ECG_S_Peaks'}},
    # T-волна
    {'name': 'T', 'points': {'start': 'ECG_T_Onsets','peak': 'ECG_T_Peaks','end': 'ECG_T_Offsets'}}
]


def calc_hrv_features(cleaned_signal, fs, waves_peak_info=None, avg_signal=False):
    if waves_peak_info is None:
        peaks, _ = nk.ecg_peaks(cleaned_signal, sampling_rate=fs)
    else:
        peaks = waves_peak_info['ECG_R_Peaks'] 
    
    hrv = pd.Series(dtype=float)
    try:
        hrv = nk.hrv(peaks, sampling_rate=fs, show=False).loc[0]
    except Exception as e:
        print(f"Error in hrv: {e}")

    return hrv

def calc_hrv_frequency_features(cleaned_signal, fs, waves_peak_info=None, avg_signal=False, method="lomb"):
    if waves_peak_info is None:
        peaks, _ = nk.ecg_peaks(cleaned_signal, sampling_rate=fs)
    else:
        peaks = waves_peak_info['ECG_R_Peaks']
    
    hrv_freq = pd.Series(dtype=float)
    
    try:
        hrv_freq = nk.hrv_frequency(
            peaks,
            sampling_rate=fs,
            psd_method=method,
            show=False
        ).loc[0]
        print(hrv_freq)
        #if len(peaks["ECG_R_Peaks"]) < 15:  # ~10 RR-интервалов для 10 сек
        #    hrv_freq[["HRV_ULF", "HRV_VLF", "HRV_LF"]] = np.nan
        #    hrv_freq["HRV_LFHF"] = np.nan
            
    except Exception as e:
        print(f"Error in HRV frequency analysis: {e}")
        """hrv_freq = pd.Series({
            "HRV_ULF": np.nan,
            "HRV_VLF": np.nan,
            "HRV_LF": np.nan,
            "HRV_HF": np.nan,
            "HRV_VHF": np.nan,
            "HRV_TP": np.nan,
            "HRV_LFHF": np.nan,
            "HRV_LFn": np.nan,
            "HRV_HFn": np.nan,
            "HRV_LnHF": np.nan
        })"""
    
    return hrv_freq

def calc_hrv_freq_hf_features(cleaned_signal, fs, waves_peak_info=None, avg_signal=False):
    if waves_peak_info is None:
        peaks, _ = nk.ecg_peaks(cleaned_signal, sampling_rate=fs)
        r_peaks = peaks['ECG_R_Peaks']
    else:
        r_peaks = waves_peak_info['ECG_R_Peaks']
    
    try:
        rr_intervals = np.diff(r_peaks) / fs
        t = np.cumsum(rr_intervals)
        hrv_freq = pd.Series(dtype=float)

        hf_range = (0.15, 0.4)  # Hz
        frequencies = np.linspace(hf_range[0], hf_range[1], 100)
        power = LombScargle(t, rr_intervals).power(frequencies)
        hf_power = np.trapz(power, frequencies)
        hrv_freq['cust_HF'] = hf_power
    except Exception as e:
        print(f"Error in HF calculation: {e}")
    return hrv_freq



def calc_wavelet_features(cleaned_signal, fs, waves_peak_info, avg_signal=False):
    coeffs = pywt.wavedec(cleaned_signal, 'db4', level=5)
    energy_d3 = np.sum(coeffs[-3]**2)  # QRS-комплекс (D3 ≈ 25-50 Гц)
    energy_d4 = np.sum(coeffs[-2]**2)  # ST-T сегмент (D4 ≈ 12.5-25 Гц)
    
    # 2. Патологические маркеры
    ## 2.1 Для ишемии/гипоксии
    st_depression = np.mean(coeffs[-2][coeffs[-2] < 0])  # Отрицательные коэффициенты D4
    t_wave_asymmetry = stats.skew(coeffs[-2])            # Асимметрия T-волны
    ## 2.2 Для аритмий
    qrs_std = np.std(coeffs[-3])  # Разброс QRS (желудочковые экстрасистолы)
    # 3. Энтропийные меры
    entropy_d4 = sample_entropy(coeffs[-2], order=2, metric='chebyshev') #"D4 (ST-T)"
    features = pd.Series({
        'energy_d3': energy_d3,
        'energy_d4': energy_d4,
        'qrs_t_ratio': energy_d3 / (energy_d4 + 1e-6),
        'st_depression': st_depression,
        't_wave_asymmetry': t_wave_asymmetry,
        'qrs_std': qrs_std,
        'entropy_d4': entropy_d4,
        #'wavelet_type': 'db4',
        #'levels': 'D1-D5 (100-6.25 Гц)'
    })
    
    return features

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

def calc_wave_asymmetry_features(cleaned_signal, fs, waves_peak_info, avg_signal=False, element_config=element_config):
    features = pd.Series(dtype=float)

    waves_peak_info = prepare_wave_data(waves_peak_info)
    grouped_cycles = group_pqrst_points(waves_peak_info)

    all_ratios = {elem['name']: [] for elem in element_config}
    for cycle in grouped_cycles:
        for elem in element_config:
            if elem['name'] == "QRS" or elem['name'] == "S":
                continue
            if not all(p in cycle for p in [elem['points']['start'], elem['points']['peak'], elem['points']['end']]):
                continue
            start = int(cycle[elem['points']['start']][0])
            peak = int(cycle[elem['points']['peak']][0])
            end = int(cycle[elem['points']['end']][0])

            segment = cleaned_signal[start:end]
            if len(segment) < 2:
                continue
            
            seg_min, seg_max = np.min(segment), np.max(segment)
            if np.isclose(seg_max, seg_min):
                segment_norm = np.zeros_like(segment)
            else:
                segment_norm = (segment - seg_min) / (seg_max - seg_min + 1e-6)
            
            peak_pos = peak - start
            area_before = np.trapz(segment_norm[:peak_pos])
            area_after = np.trapz(segment_norm[peak_pos:])
            all_ratios[elem['name']].append(area_before / (area_after + 1e-6))
    
    for wave_name, ratios in all_ratios.items():
        features[f"{wave_name}_Area_Ratio_mean"] = np.mean(ratios)
        features[f"{wave_name}_Area_Ratio_std"] = np.std(ratios)
        features[f"{wave_name}_Area_Ratio_min"] = np.min(ratios)
        features[f"{wave_name}_Area_Ratio_max"] = np.max(ratios)
        features[f"{wave_name}_Area_Ratio_median"] = np.median(ratios)
        features[f"{wave_name}_Area_Ratio_q25"] = np.quantile(ratios, 0.25)
        features[f"{wave_name}_Area_Ratio_q75"] = np.quantile(ratios, 0.75)

    return features

def calc_derivative_features(cleaned_signal, fs, waves_peak_info, avg_signal=False, element_config=element_config):
    waves_peak_info = prepare_wave_data(waves_peak_info)
    grouped_cycles = group_pqrst_points(waves_peak_info)

    deriv1 = savgol_filter(cleaned_signal, window_length=int(0.02*fs), polyorder=3, deriv=1)
    deriv2 = savgol_filter(cleaned_signal, window_length=int(0.03*fs), polyorder=3, deriv=2)

    feature_values = defaultdict(list)
    for cycle in grouped_cycles:
        for elem in element_config:
            required_points = list(elem['points'].values())
            if not all(p in cycle for p in required_points):
                continue
            start = int(cycle[elem['points']['start']][0])
            peak = int(cycle[elem['points']['peak']][0])
            end = int(cycle[elem['points']['end']][0])

            for point_name, point in [('start', start), ('peak', peak), ('end', end)]:
                if point < len(deriv1):
                    angle = np.degrees(np.arctan(deriv1[point]))
                    feature_values[f"{elem['name']}_{point_name}_angle"].append(angle)
                    #print(f"{elem['name']}_{point_name}_angle", feature_values[f"{elem['name']}_{point_name}_angle"])
            segment_start = min(start, end)
            segment_end = max(start, end)
            seg_deriv1 = deriv1[segment_start:segment_end]
            seg_deriv2 = deriv2[segment_start:segment_end]

            feature_values[f"{elem['name']}_Max_Slope"].append(np.max(seg_deriv1))
            feature_values[f"{elem['name']}_Min_Slope"].append(np.min(seg_deriv1))
            feature_values[f"{elem['name']}_Max_Convexity"].append(np.max(seg_deriv2))
            feature_values[f"{elem['name']}_Min_Concavity"].append(np.min(seg_deriv2))
    #print(feature_values)
    features = calculate_feature_statistics(feature_values, prefix="deriv_")
    return features

def calc_ecg_angles_features(cleaned_signal, fs, waves_peak_info, avg_signal=False):
    """
    Вычисляет углы наклона между ключевыми точками ЭКГ для каждого кардиоцикла.
    Возвращает статистику по всем циклам (среднее, квантили и др.).
    Основано на:
    Pal, Anita, and Yogendra Narain Singh. 
    "Biometric recognition using area under curve analysis of electrocardiogram." 
    International Journal of Advanced Computer Science and Applications 10.1 (2019).
    """
    waves_peak_info = prepare_wave_data(waves_peak_info)
    grouped_cycles = group_pqrst_points(waves_peak_info)
    
    angle_config = [
        {'name': 'Angle_P', 'start': 'ECG_P_Onsets', 'end': 'ECG_P_Offsets'},
        {'name': 'Angle_Q', 'start': 'ECG_P_Peaks', 'end': 'ECG_R_Peaks'},
        {'name': 'Angle_R', 'start': 'ECG_R_Onsets', 'end': 'ECG_R_Offsets'},
        {'name': 'Angle_QRS', 'start': 'ECG_Q_Peaks', 'end': 'ECG_S_Peaks'},
        {'name': 'Angle_S', 'start': 'ECG_R_Peaks', 'end': 'ECG_T_Peaks'},
        {'name': 'Angle_T', 'start': 'ECG_T_Onsets', 'end': 'ECG_T_Offsets'}
    ]
    
    angle_values = defaultdict(list)    
    for cycle in grouped_cycles:
        for angle in angle_config:
            if angle['start'] not in cycle or angle['end'] not in cycle:
                continue
                
            try:
                p1 = int(np.round(cycle[angle['start']][0]))
                p2 = int(np.round(cycle[angle['end']][0]))
                
                dx = (p2 - p1) / fs
                dy = cleaned_signal[p2] - cleaned_signal[p1]
                angle_deg = np.degrees(np.arctan2(dy, dx)) if dx != 0 else 90.0
                angle_values[angle['name']].append(angle_deg)
                
            except (IndexError, ValueError):
                continue
    
    features = calculate_feature_statistics(angle_values)
    return features

