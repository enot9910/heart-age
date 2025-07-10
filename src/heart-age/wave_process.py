import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from cycles_signal_process import calc_nan_wave_data

def get_waves_quality_stats(
    processed_dir, 
    dir_peaks='ptb_xl_peaks', 
    method='dwt', 
    output_file='ptb_xl_waves_quality_stats.parquet',
    channel_names=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
):
    stats_df = pd.DataFrame()
    
    peaks_dir = processed_dir / dir_peaks / method
    npz_files = list(peaks_dir.glob("*.npz"))
    
    if not npz_files:
        raise ValueError(f"Не найдено файлов .npz в директории {peaks_dir}")
    
    for file_path in tqdm(npz_files, desc="Analyzing waves quality"):
        try:
            loaded_waves_peak = np.load(file_path, allow_pickle=True)
            waves_peak_info = {
                key: loaded_waves_peak[key].item() for key in loaded_waves_peak if not key.startswith('__')
            }
            
            wave_stats = calc_nan_wave_data(waves_peak_info, channel_names)
            patient_id = file_path.stem.split('_')[1]
            wave_stats['patient_id'] = patient_id
            
            wave_stats_df = pd.DataFrame([wave_stats])
            stats_df = pd.concat([stats_df, wave_stats_df], ignore_index=True)
            
        except Exception as e:
            print(f"Ошибка при обработке файла {file_path.name}: {e}")
    
    output_dir = processed_dir / 'ptb_xl_peaks_info' / method
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_df.to_parquet(output_dir / output_file, index=False)
    
    return stats_df
