from cycles_signal_process import (
    calc_average_signal
)
from features_process import (
    get_waves_peak,
)
from pathlib import Path
import neurokit2 as nk
from tqdm import tqdm
import numpy as np

channel_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def get_age_group(age, age_groups):
    if age_groups is None:
        return age
    
    age_groups_sorted = sorted(age_groups)
    age_key = None
    
    for i in range(len(age_groups_sorted)-1):
        if age_groups_sorted[i] <= age < age_groups_sorted[i+1]:
            age_key = age_groups_sorted[i]
            break
    
    if age_key is None and age >= age_groups_sorted[-1]:
        age_key = age_groups_sorted[-1]
    
    return age_key


def get_average_signals_by_age(
    processed_dir, 
    fs=500,
    npz_files=None,
    output_dir='ptb_xl_average_signals',
    age_groups=None,
    method="dwt",
    output_dir_peaks='ptb_xl_peaks', 
    show_plot=False,
    calc_waves_peak=False
):
    if npz_files is None:
        npz_files = list(processed_dir.glob("*.npz"))
        print(f"Found {len(npz_files)} patient files")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    age_data = {}
    
    for file_path in tqdm(npz_files, desc="Processing patients"):
        output_dir_peaks = processed_dir.parent / output_dir_peaks / method
        try:
            data = np.load(file_path, allow_pickle=True)
            signal = data['signal']
            patient_id = int(data['patient_id'].item())
            age = int(data['age'].item()) if 'age' in data else None
            
            if age is None:
                continue

            age_key = get_age_group(age, age_groups)
            if age_key is None:
                continue
                
            if age_key not in age_data:
                age_data[age_key] = {channel: [] for channel in channel_names}
            
            waves_peak_filename = output_dir_peaks / f"{file_path.stem}_features.npz" #TODO _features -> _peaks
            if waves_peak_filename.exists() and calc_waves_peak==False:
                loaded_waves_peak = np.load(waves_peak_filename, allow_pickle=True)
                #waves_peak_info = {key: loaded_waves_peak[key] for key in loaded_waves_peak.files}
                waves_peak_info = {
                    key: loaded_waves_peak[key].item() for key in loaded_waves_peak if not key.startswith('__')
                }
            for i, channel_name in enumerate(channel_names):
                lead_signal = signal[i]
                signal_cleaned = nk.ecg_clean(lead_signal, sampling_rate=fs)
                
                waves_peak = get_waves_peak(signal_cleaned, fs, method=method)
                r_peaks = waves_peak['ECG_R_Peaks']
                
                if len(r_peaks) > 0:
                    avg_signal, _, _ = calc_average_signal(signal_cleaned, r_peaks, fs)
                    age_data[age_key][channel_name].append(avg_signal)
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return age_data


def save_age_group_data(age_data, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez(output_dir / "age_group_info.npz", age_data=age_data)
    
    for age, channels_data in age_data.items():
        age_dir = output_dir / str(age)
        age_dir.mkdir(exist_ok=True)
        
        for channel, signals in channels_data.items():
            if len(signals) > 0:
                final_avg_signal = np.mean(signals, axis=0)
                np.savez(
                    age_dir / f"{channel}_avg.npz",
                    signal=final_avg_signal,
                    age=age,
                    channel=channel,
                    num_patients=len(signals)
                )
    
    return output_dir