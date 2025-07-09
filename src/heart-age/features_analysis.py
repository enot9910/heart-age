
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def calculate_age_correlations(
    df: pd.DataFrame,
    age_column: str = 'age',
    save_dir: str = 'scatter_plots',
    exclude_columns: set = {'patient_id', 'target', 'original_length', 'file_name'},
    min_samples: int = 100
) -> pd.DataFrame:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    features = [col for col in numeric_cols 
               if col not in exclude_columns and col != age_column]
    
    corr_df = pd.DataFrame(index=features, columns=['correlation', 'n_samples'])
    
    age_values = df[age_column].values
    for feature in features:
        feature_values = df[feature].values
        valid_mask = ~np.isnan(age_values) & ~np.isnan(feature_values)
        n_valid = valid_mask.sum()
        
        if n_valid >= min_samples:
            corr = np.corrcoef(age_values[valid_mask], feature_values[valid_mask])[0, 1]
            corr_df.loc[feature, 'correlation'] = corr
            corr_df.loc[feature, 'n_samples'] = n_valid
    
    corr_df = corr_df.dropna(subset=['correlation'])
    
    corr_df['abs_correlation'] = corr_df['correlation'].abs()
    corr_df = corr_df.sort_values('abs_correlation', ascending=False)
    corr_df = corr_df.drop(columns=['abs_correlation'])
    
    corr_df.to_csv(save_dir / 'feature_age_correlations.csv', index=True, index_label='feature')
    
    return corr_df

def plot_features_vs_age(
    df: pd.DataFrame,
    age_column: str = 'age',
    save_dir: str = 'scatter_plots',
    exclude_columns: set = {'patient_id', 'target', 'original_length', 'file_name'}
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    features = [col for col in df.columns if col not in exclude_columns]

    for feature_name in features:
        plt.figure(figsize=(8, 6))
        
        # Create scatter plot with regression line
        sns.regplot(
            x=df[age_column],
            y=df[feature_name],
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red'}
        )
        
        valid_data = df[[age_column, feature_name]].dropna()
        corr = np.corrcoef(valid_data[age_column].values, valid_data[feature_name].values)[0, 1]
        
        plt.title(f'{feature_name} vs Age (Corr: {corr:.2f})')
        plt.xlabel('Age')
        plt.ylabel(feature_name)
        plt.grid(True)

        save_path = save_dir / f"{feature_name}_vs_age.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()