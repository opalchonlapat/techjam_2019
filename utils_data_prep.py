import pandas as pd
from sklearn.model_selection import train_test_split

def split_validation(X: pd.DataFrame, y: pd.Series, num_bin: int = 50, test_size: float = 0.2, seed: int = 42) -> list:
    """
        Splitting dataset using bins for balancing continuous target variable

        Output
        --------
        X_train, X_test, y_train, y_test
    """
    bins = pd.cut(y, num_bin, labels = [str(i) for i in range(num_bin)])
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=bins)