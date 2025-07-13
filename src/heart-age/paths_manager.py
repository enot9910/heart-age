from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional, Union
from tqdm import tqdm


def get_features_paths( # TODO class
    processed_dir: Union[str, Path],
    dataset: str,
    features_to_load: List[str] = [
        'subject_info', 'waves_quality', 
        'signal_morphology', 'signal_hrv',
        'signal_assymetry', 'signal_angles', 'signal_deriv', 
    ],
    method: str = "dwt",
) -> Dict[str, Path]:
    processed_dir = Path(processed_dir)
    paths = {}
    
    features_paths = {
        'signal_morphology': {
            'dir': processed_dir / f"{dataset}_features_signal_morphology" / method,
            'pattern': "patient_features_*.parquet"
        },
        'signal_morphology_avg': {
            'dir': processed_dir / f"{dataset}_features_signal_morphology_avg" / method,
            'pattern': "patient_features_*.parquet"
        },
        'signal_assymetry': {
            'dir': processed_dir / f"{dataset}_features_signal_asymmetry" / method,
            'pattern': "patient_features_*.parquet"
        },
        'signal_deriv': {
            'dir': processed_dir / f"{dataset}_features_signal_deriv" / method,
            'pattern': "patient_features_*.parquet"
        },
        'signal_hrv': {
            'dir': processed_dir / f"{dataset}_features_signal_hrv" / method,
            'pattern': "patient_features_*.parquet"
        },
        'signal_angles': {
            'dir': processed_dir / f"{dataset}_features_signal_angles" / method,
            'pattern': "patient_features_*.parquet"
        },
        'subject_info': {
            'dir': processed_dir / f"{dataset}_subject_info",
            'pattern': "subject_info_all.parquet"
        },
        'waves_quality': {
            'dir': processed_dir / f"{dataset}_peaks_info" / method,
            'pattern': "ptb_xl_waves_quality_stats.parquet"
        }
    }
    
    for feature in features_to_load:
        if feature not in features_paths:
            raise ValueError(f"Unknown feature type: {feature}. Available options: {list(features_paths.keys())}")
            
        config = features_paths[feature]
        files = list(config['dir'].glob(config['pattern']))
        
        if not files:
            raise FileNotFoundError(f"No files found for {feature} at {config['dir']}\{config['pattern']}")
            
        paths[feature] = files[0] if len(files) == 1 else files
    
    return paths
