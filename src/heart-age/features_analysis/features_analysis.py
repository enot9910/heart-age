
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

def calculate_age_correlations(
    df: pd.DataFrame,
    age_column: str = 'age',
    exclude_columns: set = {'patient_id', 'target', 'original_length', 'file_name'},
    min_samples: int = 100,
    alpha: float = 0.05,
    save_corr: bool = False,
    save_dir: str = '.',
) -> pd.DataFrame:
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    features = [col for col in numeric_cols 
                if col not in exclude_columns and col != age_column]
    
    corr_df = pd.DataFrame(
        index=features,
        columns=[
            'correlation', 'p_value', 'p_value_bonferroni', 
            'significant','regression_slope','n_samples'
        ]
    )
    
    age_values = df[age_column].values.reshape(-1, 1)
    n_features = len(features)
    
    for feature in features:
        feature_values = df[feature].values
        valid_mask = ~np.isnan(age_values.flatten()) & ~np.isnan(feature_values)
        n_valid = valid_mask.sum()
        
        if n_valid >= min_samples:
            corr, p_value = spearmanr(age_values[valid_mask].flatten(), feature_values[valid_mask])
            p_value_bonferroni = min(p_value * n_features, 1.0)
            is_significant = p_value_bonferroni < alpha
            
            lr = LinearRegression()
            lr.fit(age_values[valid_mask], feature_values[valid_mask])
            slope = lr.coef_[0]
            corr_df.loc[feature, 'correlation'] = corr
            corr_df.loc[feature, 'p_value'] = p_value
            corr_df.loc[feature, 'p_value_bonferroni'] = p_value_bonferroni
            corr_df.loc[feature, 'significant'] = is_significant
            corr_df.loc[feature, 'regression_slope'] = slope
            corr_df.loc[feature, 'n_samples'] = n_valid
    
    corr_df = corr_df.dropna(subset=['correlation'])
    corr_df['abs_correlation'] = corr_df['correlation'].abs()
    corr_df = corr_df.sort_values('abs_correlation', ascending=False)
    corr_df = corr_df.drop(columns=['abs_correlation'])
    
    if save_corr:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
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

from sklearn.feature_selection import VarianceThreshold
def filtered_variance_threshold(
    df: pd.DataFrame,
    percentile_range: tuple = (0.2, 99.7),
    threshold: float = 0.0
) -> pd.DataFrame:
    filtered_df = df.copy()
    
    for column in df.select_dtypes(include=[np.number]).columns:
        low = np.nanpercentile(df[column], percentile_range[0])
        high = np.nanpercentile(df[column], percentile_range[1])
        filtered_df.loc[(df[column] < low) | (df[column] > high), column] = np.nan
    
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(filtered_df)

    variances = selector.variances_
    variance_df = pd.DataFrame({'feature': df.columns, 'variance': variances})
    variance_df = variance_df.sort_values('variance', ascending=False)

    selected_features = df.columns[selector.get_support()]
    del_features = df.columns[~selector.get_support()]
    
    return selected_features, del_features, variance_df