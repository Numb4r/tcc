import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from sklearn.model_selection import train_test_split


def cleaning_columns(data: pd.DataFrame, keep: list() = [], remove: list = []):

    if keep is not None or len(keep) != 0:
        data = data[[*keep]]
    if remove is not None or len(remove) != 0:
        for k, v in enumerate(remove):
            data.drop(v, axis=1)
    return data


def split_train_test(data, results, test_size=0.10, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(
        data, results, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


# # Ajuste o scaler aos dados e transforme as colunas selecionadas


def get_outliers(df: pd.DataFrame, z_score_threshold: int = 3):

    # Defina um limiar para identificar outliers (por exemplo, 3)
    # z_score_threshold = 3
    # Calcule os escores Z para todas as colunas
    z_scores = stats.zscore(df)

    # Identifique outliers em todas as colunas
    outliers_indices = (abs(z_scores) > z_score_threshold).any(axis=1)

    # Crie um DataFrame booleano indicando a presença de outliers em cada célula
    # outliers_df = pd.DataFrame(outliers, columns=df.columns)
    # outlier_indices = outliers.any(axis=1).nonzero()[0]
    # # Exiba o DataFrame com informações sobre os outliers
    return df[outliers_indices].index


def column_divisor(x, numerator: list, denominator: int):
    for n in numerator:
        x[n] = x[n] / x[denominator]
    return x


def normalize_columns(data):
    variables_columns = ['INTEGERS', 'BINARIES', 'CONTINUOUS']
    constraints_columns = ['CONSTR.EMPTY', 'CONSTR.FREE', 'CONSTR.SINGLETON', 'CONSTR.AGGREGATION', 'CONSTR.PRECEDENCE', 'CONSTR.VARIABLE.BOUND', 'CONSTR.SET.PARTITIONING', 'CONSTR.SET.PACKING', 'CONSTR.SET.COVERING',
                           'CONSTR.BINPACKING', 'CONSTR.KNAPSACK', 'CONSTR.INTEGER.KNAPSACK', 'CONSTR.CARDINALITY', 'CONSTR.INVARIANT.KNAPSACK', 'CONSTR.EQUATION.KNAPSACK', 'CONSTR.MIXED.BINARY', 'CONSTR.GENERAL.LINEAR']
    data = column_divisor(data, variables_columns, "VARIABLES")
    data = column_divisor(data, constraints_columns, "CONSTRAINTS")
    return data
