
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold
import networkx as nx

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


def scatter_features_vs_age(
    df: pd.DataFrame,
    age_column: str = 'age',
    save_dir: str = 'scatter_plots',
    exclude_columns: set = {'patient_id', 'target', 'original_length', 'file_name'},
    features: list = None,
    alpha: float = 0.6,
    kdeplot: bool = False,
    s_scatter: float = 0.01
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if features is None:
        features = [col for col in df.columns if col not in exclude_columns and col != age_column]
    else:
        features = [
            col for col in features if col in df.columns and col not in exclude_columns and col != age_column
        ]

    for feature_name in features:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        
        if kdeplot:
            sns.kdeplot(
                x=df[age_column],
                y=df[feature_name],
                fill=True,
                cmap='Blues',
                alpha=0.8,
                ax=ax
            )
            plt.colorbar(ax.collections[0])
        
        sns.regplot(
            x=df[age_column],
            y=df[feature_name],
            scatter_kws={'alpha': alpha, 's': s_scatter, 'color': 'red'},
            line_kws={'color': 'red', 'linewidth': 2},
            ax=ax
        )
        
        valid_data = df[[age_column, feature_name]].dropna()
        corr = np.corrcoef(valid_data[age_column].values, valid_data[feature_name].values)[0, 1]
        parts = feature_name.split('_')
        filtered_parts = [p for p in parts if 'ECG' not in p]
        display_name = ' '.join(filtered_parts[:-2]) if len(filtered_parts) > 2 else ' '.join(filtered_parts)
        
        plt.title(f'{display_name} vs Age (Corr: {corr:.2f})', fontsize=14)
        plt.xlabel('Age', fontsize=12)
        plt.ylabel(display_name, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        save_path = save_dir / f"{feature_name}_vs_age.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

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

def calculate_feature_correlations(
    df: pd.DataFrame,
    exclude_columns: set = None,
    correlation_threshold: float = 0.9,
    min_samples: int = 100
) -> pd.DataFrame:
    if exclude_columns is None:
        exclude_columns = set()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    features = [col for col in numeric_cols if col not in exclude_columns]
    df_subset = df[features]
    corr_matrix = df_subset.corr(method='pearson', min_periods=min_samples)
    
    corr_pairs = (
        corr_matrix.stack()
        .reset_index()
        .rename(columns={'level_0': 'feature1', 'level_1': 'feature2', 0: 'correlation'})
    )
    
    corr_pairs = corr_pairs[corr_pairs['feature1'] < corr_pairs['feature2']]
    high_corr_pairs = corr_pairs[abs(corr_pairs['correlation']) >= correlation_threshold]
    
    results = []
    for _, row in high_corr_pairs.iterrows():
        feat1, feat2 = row['feature1'], row['feature2']
        valid_mask = ~df[feat1].isna() & ~df[feat2].isna()
        n_valid = valid_mask.sum()
        
        _, p_value = pearsonr(df.loc[valid_mask, feat1], df.loc[valid_mask, feat2])
        
        results.append({
            'feature1': feat1,
            'feature2': feat2,
            'correlation': row['correlation'],
            'p_value': p_value,
            'n_samples': n_valid
        })
    
    return pd.DataFrame(results)



def get_best_feature(features_group, target_correlations, p_value_threshold=0.5):
    if target_correlations is None or target_correlations.empty:
        return next(iter(features_group))

    valid_features = [
        feature for feature in features_group
        if feature in target_correlations.index
    ]

    if 'p_value' in target_correlations.columns:
        valid_features = [
            feature for feature in valid_features
            if target_correlations.loc[feature, 'p_value'] < p_value_threshold
        ]

    best_feature = max(
        valid_features,
        key=lambda x: abs(target_correlations.loc[x, 'correlation'])
    )
    return best_feature

def select_best_features(
    corr_pairs,
    target_correlations=None,
    correlation_threshold=1.0,
    p_value_threshold=0.5
):
    unique_features = set(corr_pairs['feature1']).union(set(corr_pairs['feature2']))

    G = nx.Graph()
    for _, row in corr_pairs.iterrows():
        if abs(row['correlation']) >= correlation_threshold:
            G.add_edge(row['feature1'], row['feature2'])


    connected_components = list(nx.connected_components(G))
    print(f"\nГруппы коррелирующих фич (corr >= {correlation_threshold}):")
    for i, group in enumerate(connected_components, 1):
        print(f"Группа {i}: {group}")

    features_to_keep = []
    for group in connected_components:
        best_feature = get_best_feature(group, target_correlations, p_value_threshold)
        if best_feature is not None:
            features_to_keep.append(best_feature)

    independent_features = unique_features - set(G.nodes())
    if target_correlations is not None:
        independent_features = [
            feature for feature in independent_features
            if feature in target_correlations.index and (
                'p_value' not in target_correlations.columns or
                target_correlations.loc[feature, 'p_value'] < p_value_threshold
            )
        ]
    features_to_keep.extend(independent_features)
    print(f"Итоговое количество: {len(features_to_keep)}")
    return features_to_keep

def find_inf_columns(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
    
    inf_cols = []
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            inf_count = np.isinf(df[col]).sum()
            print(f"Колонка {col}: {inf_count} inf/-inf значений")
            inf_cols.append(col)
    
    return inf_cols