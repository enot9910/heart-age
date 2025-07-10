from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional, Union
from tqdm import tqdm



def merge_features(
    features_paths: Dict[str, Union[Path, List[Path]]],
    merge_on: str = "patient_id",
    no_suffix_cols: List[str] = ["target", "sex", "age", "file_name"],
    how: str = "outer"
) -> pd.DataFrame:
    merged_df = None
    no_suffix_cols = list(set(no_suffix_cols + [merge_on]))
    
    for feature_type, paths in features_paths.items():
        if not isinstance(paths, list):
            paths = [paths]
            
        dfs = []
        for path in paths:
            try:
                df = pd.read_parquet(path)
                df[merge_on] = df[merge_on].astype(int)
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
                
        if not dfs:
            continue

        feature_df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

        cols_to_suffix = [col for col in feature_df.columns if col not in no_suffix_cols]
        suffix_mapping = {col: f"{col}_{feature_type}" for col in cols_to_suffix}
        feature_df = feature_df.rename(columns=suffix_mapping)

        if merged_df is None:
            merged_df = feature_df
        else:
            existing_cols = set(merged_df.columns)
            new_cols = [col for col in feature_df.columns if col not in existing_cols]
            cols_to_merge = [merge_on] + new_cols
            
            merged_df = pd.merge(
                merged_df,
                feature_df[cols_to_merge],
                on=merge_on,
                how=how
            )
    
    return merged_df.drop_duplicates()