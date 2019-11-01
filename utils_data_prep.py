import numpy as np
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

def read_given_files() -> list:
    """
        Read given KBTG files and define data type to appropriate type

        Output
        ---------
        demographic_df, credit_df, kplus_df, train_df
    """
    demographic_df = pd.read_csv('/content/drive/My Drive/techjam_2019/data_pack/demographics.csv', 
                                dtype={'id':str, 'cc_no':str, 'gender':'category', 'ocp_cd':'category','age':'category'})
    credit_df = pd.read_csv('/content/drive/My Drive/techjam_2019/data_pack/cc.csv',
                            dtype={'cc_no':str}, parse_dates=['pos_dt'])
    kplus_df = pd.read_csv('/content/drive/My Drive/techjam_2019/data_pack/kplus.csv',
                        dtype={'id':str}, parse_dates=['sunday'])
    train_df = pd.read_csv('/content/drive/My Drive/techjam_2019/data_pack/train.csv',
                        dtype={'id':str})
    return demographic_df, credit_df, kplus_df, train_df

def modified_smape(y_true, y_pred):
    return (100 - 100/len(y_pred) * 
            np.sum(np.abs(y_pred - y_true) ** 2 /
                (np.min(np.abs(np.concatenate([y_pred.reshape(-1,1), 2 * y_true.reshape(-1,1)],1)),1) + np.abs(y_true)) ** 2))
