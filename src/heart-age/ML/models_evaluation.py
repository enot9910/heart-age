import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

def train_evaluate(
    X_train, y_train, 
    col, clf, 
    random_number, param_grid={},
    bins=[-1, 30, 40, 50, 60, 70, 85], 
    labels=None, refit_full_model=True
):
    import warnings
    warnings.filterwarnings("ignore")

    if labels is None:
        labels = list(range(len(bins) - 1))
    valid_mae_list = []
    train_mae_list = []
    y = pd.cut(y_train,  bins=bins, labels=labels)

    if param_grid:
        print('GridSearchCV')
        print(clf, param_grid, skf.split(X_train[col], y))
        grid_search = GridSearchCV(clf, param_grid, scoring='neg_mean_absolute_error', n_jobs=-1, cv=skf.split(X_train[col], y)) #
        grid_search.fit(X_train[col], y_train)
        clf = grid_search.best_estimator_
    models = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_number)
    for train_index, valid_index in skf.split(X_train[col], y): #, y_train
        X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[valid_index]
        y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]
        X_train_fold, X_valid_fold = X_train_fold[col], X_valid_fold[col]
        clf = clf.fit(X_train_fold, y_train_fold, eval_set=[(X_valid_fold, y_valid_fold)])

        y_pred_valid = clf.predict(X_valid_fold)
        y_pred_train = clf.predict(X_train_fold)
        valid_mae = mean_absolute_error(y_valid_fold, y_pred_valid)
        train_mae = mean_absolute_error(y_train_fold, y_pred_train)

        valid_mae_list.append(valid_mae)
        train_mae_list.append(train_mae)
        models.append(clf)

    mean_valid_mae = np.mean(valid_mae_list)
    mean_train_mae = np.mean(train_mae_list)
    final_model = clone(clf)
    final_model.fit(X_train[col], y_train)

    print(f"Модель: {final_model}")
    print(f"Mean Valid MAE: {mean_valid_mae:.4f}")
    print(f"Mean Train MAE: {mean_train_mae:.4f}")
    return final_model, mean_valid_mae, mean_train_mae, random_number


def evaluate(y_true, y_pred, dataset_name="Данные"):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)
    
    print(f"{dataset_name}: MAE: {mae:.4f}, R^2: {r2:.4f}, corr(Пирсона): {corr:.4f}\n")

def evaluate_group_results(y_true_train, y_pred_train, y_true_test, y_pred_test, X_train, X_test):
    test_results = pd.DataFrame({
        'y_true': y_true_test,
        'y_pred': y_pred_test
    }, index=X_test.index)

    train_results = pd.DataFrame({
        'y_true': y_true_train,
        'y_pred': y_pred_train
    }, index=X_train.index)

    median_by_group_train = train_results.groupby(level=0).median()
    median_by_group_test = test_results.groupby(level=0).median()
    evaluate(
        median_by_group_train['y_true'], median_by_group_train['y_pred'], dataset_name="Train"
    )
    evaluate(median_by_group_test['y_true'], median_by_group_test['y_pred'], dataset_name="Test")

from sklearn.metrics import median_absolute_error

def plot_predicted_scatter(axes, real, pred, xlabel='Age', ylabel='Pred Age', title='', fontsize=15, dfontsize=1):
    sns.scatterplot(x=real, y=pred, alpha=0.9, s=15, ax=axes)
    sns.regplot(x=real, y=pred, ax=axes, scatter=False, color='red', label='Линия регрессии')
    axes.plot([min(real),max(real)], [min(real),max(real)], color="grey", linestyle='--')

    mae = mean_absolute_error(real, pred)
    medae = median_absolute_error(real, pred)
    r2 = r2_score(real, pred)
    corr, _ = pearsonr(real, pred)
    
    metrics_text = f"{title}: MAE={mae:.2f}, MedAE={medae:.2f}"#, R²={r2:.2f}, corr={corr:.2f}"
    axes.set_title(metrics_text, fontsize=fontsize)
    #axes.text(0.05, 0.86, f'corr: {corr_test:.2f}', transform=axes.transAxes, fontsize=fontsize-dfontsize)
    #axes.text(0.05, 0.95, f'MAE: {mae:.2f}', transform=axes.transAxes, fontsize=fontsize-dfontsize)
    
    axes.set_xlabel(xlabel, fontsize=fontsize, fontweight='light')
    axes.set_ylabel(ylabel, fontsize=fontsize, fontweight='light')
    axes.tick_params(axis='x', labelsize=fontsize-dfontsize)
    axes.tick_params(axis='y', labelsize=fontsize-dfontsize)
    axes.legend(fontsize=fontsize-2*dfontsize, loc='upper right')
    axes.grid(True)
