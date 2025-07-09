import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


from cycles_signal_process import calc_nan_wave_data
def get_waves_quality_stats(processed_dir, output_dir_peaks='ptb_xl_peaks', method='dwt', output_file='ptb_xl_waves_quality_stats.parqet'):
    stats_df = pd.DataFrame()
    
    peaks_dir = processed_dir / output_dir_peaks / method
    npz_files = list(peaks_dir.glob("*.npz"))
    
    if not npz_files:
        raise ValueError(f"Не найдено файлов .npz в директории {peaks_dir}")
    
    for file_path in tqdm(npz_files, desc="Analyzing waves quality"):
        try:
            loaded_waves_peak = np.load(file_path, allow_pickle=True)
            waves_peak_info = {
                key: loaded_waves_peak[key].item() for key in loaded_waves_peak if not key.startswith('__')
            }
            
            wave_stats = calc_nan_wave_data(waves_peak_info)
            patient_id = file_path.stem.split('_')[1]
            wave_stats['patient_id'] = patient_id
            
            wave_stats_df = pd.DataFrame([wave_stats])
            stats_df = pd.concat([stats_df, wave_stats_df], ignore_index=True)
            
        except Exception as e:
            print(f"Ошибка при обработке файла {file_path.name}: {e}")
    
    output_path = processed_dir.parent / output_file
    stats_df.to_parquet(output_path, index=False)
    
    return stats_df
