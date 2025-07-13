import os
import numpy as np
from tqdm import tqdm
from ecglib.preprocessing import BaselineWanderRemoval, IIRNotchFilter, ButterworthFilter
from pathlib import Path

class EcgFiltering:
    def __init__(self, fs=500):
        self.fs = fs
        
    def __call__(self, ecg_record):
        signal = ecg_record
        
        baseline_filter = BaselineWanderRemoval()
        signal = baseline_filter(signal)
        notch_filter = IIRNotchFilter(w0=50, Q=30, fs=self.fs)
        signal = notch_filter(signal)
        
        bandpass_filter = ButterworthFilter(
            filter_type='bandpass',
            fs=self.fs,
            n=3,
            Wn=[0.5, 45],
        )
        signal = bandpass_filter(signal)

        return signal
    

def convert_signal_to_np(sample, filtering):
    signal_tensor = sample[1][0][0]
    signal_np = signal_tensor.numpy()
    processed_signal = filtering(signal_np)
    target = sample[1][1].item()
    
    return processed_signal, target

def preprocess_sample(idx, filtering):
    sample = ecg_data[idx]
    signal = sample[1][0][0].numpy()
    processed_signal = filtering(signal)
    return processed_signal

def save_npz(ecg_data, filtering, output_dir="ptb_xl_npz"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    error_indices = []
    
    for idx in tqdm(range(len(ecg_data)), desc="ЭКГ"):
        try:
            sample = ecg_data[idx]
            #print(sample, filtering)
            processed_signal, target = convert_signal_to_np(sample, filtering)
            #print(processed_signal)
            patient_id=int(ecg_data.ecg_data.iloc[idx]['patient_id'])
            file_path = output_path / f"ecg_{patient_id}.npz"
            np.savez(
                file_path,
                signal=processed_signal,
                target=target,
                patient_id=patient_id,
                age = ecg_data.ecg_data.iloc[idx]['age'],
                sex = int(ecg_data.ecg_data.iloc[idx]['sex']),
                pacemaker = ecg_data.ecg_data.iloc[idx]['pacemaker'],
                strat_fold = int(ecg_data.ecg_data.iloc[idx]['strat_fold']),
                heart_axis = ecg_data.ecg_data.iloc[idx]['heart_axis'],
                index=idx
            )            
            success_count += 1
            
        except Exception as e:
            error_indices.append(idx)
            print(f"\nОшибка при обработке записи {idx}: {str(e)}")