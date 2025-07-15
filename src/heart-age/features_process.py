import numpy as np
import pandas as pd
from pathlib import Path
import neurokit2 as nk
from tqdm import tqdm
from cycles_signal_process import calc_average_signal
from features_extraction import (
    calc_signal_morphology_features, 
    calc_hrv_features, 
    calc_ecg_angles_features,
    calc_wave_asymmetry_features,
    calc_derivative_features,
    calc_hrv_frequency_features,
    calc_hrv_freq_hf_features,
    calc_wavelet_features,
    calculate_interchannel_features
)
#from cycles_signal_process import calc_nan_wave_data

channel_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

FEATURE_EXTRACTORS = {
    'morphology': calc_signal_morphology_features,
    'hrv': calc_hrv_features,
    'angles': calc_ecg_angles_features,
    'asymmetry': calc_wave_asymmetry_features,
    'derivative': calc_derivative_features,
    'freq': calc_hrv_frequency_features,
    'HF': calc_hrv_freq_hf_features,
    'wavelet': calc_wavelet_features

}

def get_waves_peak(cleaned_signal, fs, method="dwt", waves_peak_info=None):
    if waves_peak_info is None:
        peaks, rpeaks = nk.ecg_peaks(cleaned_signal, sampling_rate=fs)
        _, waves_peak_info = nk.ecg_delineate(
            cleaned_signal, 
            rpeaks, 
            sampling_rate=fs, 
            method=method # dwt cwt peak - slow, prominence - fast
        )

        waves_peak_info['ECG_R_Peaks'] = rpeaks['ECG_R_Peaks']

    return waves_peak_info


def calc_features(
    signal_cleaned, 
    fs, 
    method="dwt", 
    waves_peak_info=None, 
    avg_signal=False, 
    show_plot=False, 
    extractors=['morphology']
):
    features_series = pd.Series(dtype=float)

    if avg_signal:
        signal, before_r, after_r = calc_average_signal(signal_cleaned, waves_peak_info['ECG_R_Peaks'], fs)
        signal_cleaned = np.tile(signal, 13)

    if show_plot:
        signals, info = nk.ecg_process(signal_cleaned, sampling_rate=fs)
        nk.ecg_plot(signals, info)

    waves_peak_info = get_waves_peak(
        signal_cleaned, fs, method=method, waves_peak_info=waves_peak_info
    )
    if extractors is None:
        return None, waves_peak_info
    
    features_series = pd.concat(
        [FEATURE_EXTRACTORS[name](signal_cleaned, fs, waves_peak_info=waves_peak_info, avg_signal=avg_signal) 
            for name in extractors],
        axis=0
    ).drop_duplicates()
    #print('features_series',features_series)
    
    return features_series, waves_peak_info

def extract_scalar(value):
    if isinstance(value, (list, np.ndarray)):
        try:
            while isinstance(value, (list, np.ndarray)):
                value = value[0]
            return value
        except IndexError:
            return np.nan
    else:
        return value 

def calc_ecg_signal_features(
    file_path, 
    fs, 
    method="dwt", 
    waves_peak_info=None, 
    avg_signal=False, 
    show_plot=False, 
    extractors=['morphology'],
    target_channel=None
):
    data = np.load(file_path, allow_pickle=True)
    signal = data['signal']
    
    features_series = pd.Series({
        'patient_id': int(data['patient_id'].item()),
    })
    waves_peak = {}
    for i, channel_name in enumerate(channel_names):
        lead_signal = signal[i]
        #print(f"\nLead {channel_name} signal length: {len(lead_signal)} samples ({len(lead_signal)/fs:.2f} seconds)")
        signal_cleaned = nk.ecg_clean(lead_signal, sampling_rate=fs)

        if waves_peak_info is None:
            waves_peak_info = waves_peak_info
        else:
            if target_channel is None:
                waves_peak_channel = waves_peak_info.get(channel_name, None)
            else:
                waves_peak_channel = waves_peak_info[target_channel]

        waves_peak_channel = waves_peak_info if waves_peak_info is None else waves_peak_info.get(channel_name, None)
        waves_peak_channel = get_waves_peak(
            signal_cleaned, fs, method=method, waves_peak_info=waves_peak_channel
        )
        channel_features, waves_peak_channel = calc_features(
            signal_cleaned, fs, method=method, waves_peak_info=waves_peak_channel, avg_signal=avg_signal, show_plot=show_plot, 
            extractors=extractors
        )
        channel_features.index = [f"{feature_name}_{channel_name}" for feature_name in channel_features.index]
        if not channel_features.empty:
            features_series = pd.concat([features_series, channel_features])
        #print(features_series)
        waves_peak[channel_name] = waves_peak_channel
    return features_series, waves_peak

def get_npz_files(input_dir): #TODO math manager
    npz_files = list(Path(input_dir).glob("*.npz"))
    print(f"Found {len(npz_files)} patient files")
    return npz_files

def save_features_batch(all_features_df, output_dir, npz_files, batch_size, idx=None):
    if idx is not None:
        if (idx + 1) % batch_size == 0 or (idx + 1) == len(npz_files):
            start_idx = (idx // batch_size) * batch_size
            end_idx = idx
            batch_filename = f"patient_features_{start_idx + 1}_to_{end_idx + 1}.parquet"
            all_features_df.to_parquet(output_dir / batch_filename, index=False)
            print(f"Saved batch {batch_filename}")
            return pd.DataFrame()
        return all_features_df
    else:
        if not all_features_df.empty:
            start_idx = (len(npz_files) // batch_size) * batch_size
            end_idx = len(npz_files) - 1
            batch_filename = f"patient_features_{start_idx + 1}_to_{end_idx + 1}.parquet"
            all_features_df.to_parquet(output_dir / batch_filename, index=False)
            print(f"Saved final batch {batch_filename}")
        return pd.DataFrame()

def prepare_output_paths(processed_dir, output_dir_features, output_dir_peaks, method, avg_signal): #TODO math manager
    output_dir_features = processed_dir.parent / f'{output_dir_features}_avg' / method \
        if avg_signal else processed_dir.parent / output_dir_features / method
    output_dir_peaks = processed_dir.parent /  f'{output_dir_peaks}_avg'  / method \
        if avg_signal else processed_dir.parent / output_dir_peaks / method
    
    output_dir_features.mkdir(parents=True, exist_ok=True)
    output_dir_peaks.mkdir(parents=True, exist_ok=True)
    
    return output_dir_features, output_dir_peaks


def get_ecg_signals_features(
    processed_dir, 
    batch_size=2000, 
    fs=500,
    npz_files=None,
    output_dir_features='ptb_xl_features_signal_morphology', 
    output_dir_peaks='ptb_xl_peaks', 
    method="dwt", 
    avg_signal=False, show_plot=False, calc_waves_peak=True, 
    extractors=['morphology'],
    target_channel=None,
    comparing_channel=False
):
    all_features_df = pd.DataFrame()

    if npz_files is None:
        npz_files = list(processed_dir.glob("*.npz"))
        print(f"Found {len(npz_files)} patient files")

    output_dir_features, output_dir_peaks = prepare_output_paths( #TODO math manager
        processed_dir, output_dir_features, output_dir_peaks, method, avg_signal
    )
    print(output_dir_features, output_dir_peaks)
    count = 0 

    for idx, file_path in enumerate(tqdm(npz_files, desc="Processing patients")):
        #if count > 2:
        #    break
        waves_peak_filename = output_dir_peaks / f"{file_path.stem}_features.npz" #TODO _features -> _peaks
        if waves_peak_filename.exists() and calc_waves_peak==False:
            loaded_waves_peak = np.load(waves_peak_filename, allow_pickle=True)
            #waves_peak_info = {key: loaded_waves_peak[key] for key in loaded_waves_peak.files}
            waves_peak_info = {
                key: loaded_waves_peak[key].item() for key in loaded_waves_peak if not key.startswith('__')
            }
        else:
            waves_peak_info=None
        try:
            if comparing_channel:
                features, waves_peak = calc_ecg_signal_features(
                    str(file_path), fs=fs, method=method, waves_peak_info=waves_peak_info, 
                    avg_signal=avg_signal, show_plot=show_plot, extractors=extractors,
                    target_channel=target_channel 
                )
            else:
                features, waves_peak = calc_ecg_signal_features_comparing(
                    str(file_path), fs=fs, method=method, waves_peak_info=waves_peak_info, 
                    avg_signal=avg_signal, show_plot=show_plot, extractors=extractors,
                    target_channel=target_channel 
                )
            if calc_waves_peak:
                np.savez(waves_peak_filename, **waves_peak)

        except Exception as e:
            print(f"Error in get_ecg_signal_features: {e}")
        features_df = features.to_frame().T
        #print("features shape:", features.shape, features.index)
        
        for col in features_df.columns:
            features_df[col] = features_df[col].apply(extract_scalar)
        all_features_df = pd.concat([all_features_df, features_df], ignore_index=True)
        #count += 1
        all_features_df = save_features_batch(all_features_df, output_dir_features, npz_files, batch_size, idx)
    all_features_df = save_features_batch(all_features_df, output_dir_features, npz_files, batch_size, idx=None)



def calc_ecg_signal_features_comparing(
    file_path, 
    fs, 
    target_channel,
    method="dwt", 
    waves_peak_info=None, 
    avg_signal=False, 
    show_plot=False, 
    extractors=['morphology'],
):
    data = np.load(file_path, allow_pickle=True)
    signal = data['signal']
    
    features_series = pd.Series({
        'patient_id': int(data['patient_id'].item()),
    })
    waves_peak = {}
    target_idx = channel_names.index(target_channel)
    target_signal_cleaned = nk.ecg_clean(signal[target_idx], sampling_rate=fs)
    waves_peak_info_target = get_waves_peak(
        target_signal_cleaned, fs, method=method, waves_peak_info=None
    )
    
    for i, channel_name in enumerate(channel_names):
        if channel_name == target_channel:
            continue
            
        lead_signal = signal[i]
        signal_cleaned = nk.ecg_clean(lead_signal, sampling_rate=fs)

        waves_peak_channel = waves_peak_info if waves_peak_info is None else waves_peak_info.get(channel_name, None)
        waves_peak_channel = get_waves_peak(
            signal_cleaned, fs, method=method, waves_peak_info=waves_peak_channel
        )
        
        channel_features, waves_peak_channel = calc_features(
            signal_cleaned, fs, method=method, waves_peak_info=waves_peak_channel, 
            avg_signal=avg_signal, show_plot=show_plot, extractors=extractors
        )
        interchannel_features = calculate_interchannel_features(
            waves_peak_info_target,
            waves_peak_channel,
            target_signal_cleaned,
            signal_cleaned,
            fs,
            target_channel,
            channel_name
        )
        channel_features = pd.concat([channel_features, interchannel_features])
        
        channel_features.index = [f"{feature_name}_{channel_name}" for feature_name in channel_features.index]
        if not channel_features.empty:
            features_series = pd.concat([features_series, channel_features])
            
        waves_peak[channel_name] = waves_peak_channel
    
    return features_series, waves_peak