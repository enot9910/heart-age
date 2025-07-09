import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def get_subject_info(file_path):
    data = np.load(file_path, allow_pickle=True)
    return pd.Series({
        'patient_id': data['patient_id'].item(),
        'target': data['target'].item() if 'target' in data else np.nan,
        'file_name': Path(file_path).name,
        'age': data['age'].item(),
        'sex': data['sex'].item(),
    })

def extract_subjects_info(processed_dir, output_dir='ptb_xl_subject_info', npz_files=None): #TODO pathmanager
    if npz_files is None:
        npz_files = list(processed_dir.glob("*.npz"))
    
    info_subject_dir = processed_dir.parent / output_dir
    info_subject_dir.mkdir(parents=True, exist_ok=True)

    all_subject_info = pd.DataFrame()
    
    for file_path in tqdm(npz_files, desc="Extracting subject info"):
        try:
            subject_info = get_subject_info(file_path)
            all_subject_info = pd.concat([all_subject_info, subject_info.to_frame().T], ignore_index=True)
        except Exception as e:
            print(f"Error processing file {file_path.name}: {e}")
    
    output_path = info_subject_dir / "subject_info_all.parqet"
    all_subject_info.to_parquet(output_path, index=False)
    
    return all_subject_info