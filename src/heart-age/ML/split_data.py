from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(
    df_dataset, 
    cols, 
    scaler,
    bins,
    target_col='age',
    test_size=0.2,
    valid_size=0.2,
    random_state=42,
    need_valid=True
):
    labels = list(range(len(bins) - 1))
    strat_groups = pd.cut(df_dataset[target_col], bins=bins, labels=labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df_dataset[cols], 
        df_dataset[target_col], 
        test_size=test_size, 
        stratify=strat_groups, 
        random_state=random_state
    )
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train = pd.DataFrame(X_train_scaled, columns=cols, index=X_train.index)
    X_test = pd.DataFrame(X_test_scaled, columns=cols, index=X_test.index)
    
    if not need_valid:
        return X_train, X_test, y_train, y_test
    
    valid_strat_groups = pd.cut(y_train, bins=bins, labels=labels)
    valid_size_adj = valid_size / (1 - test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, 
        y_train, 
        test_size=valid_size_adj, 
        stratify=valid_strat_groups, 
        random_state=random_state
    )
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test

