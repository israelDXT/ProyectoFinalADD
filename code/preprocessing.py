import argparse
import os
import requests
import tempfile

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Since we get a headerless CSV file, we specify the column names here.
feature_columns_names = ['edad', 'antiguedad_mes', 'genero', 'saldopararetiro',
                         'flag_activo', 'alta', 'tiempo_vida', 'cltv_futuro',
                         'cltv_historico', 'cltv', 'valor_12', 'flag_paperless',
                         'flag_esdigital', 'modulo', 'tienda', 'dias_cot6', 'aporp6',
                         'sbc_6', 'can_apor6']

label_column = 'flag_cedido'

feature_columns_dtype = {
    'edad': np.float64,
    'antiguedad_mes': np.float64,
    'genero': np.float64,
    'saldopararetiro': np.float64,
    'flag_activo': np.float64,
    'alta': np.float64,
    'tiempo_vida': np.float64,
    'cltv_futuro': np.float64,
    'cltv_historico': np.float64,
    'cltv': np.float64,
    'valor_12': np.float64,
    'flag_paperless': np.float64,
    'flag_esdigital': np.float64,
    'modulo': np.float64,
    'tienda': np.float64,
    'dias_cot6': np.float64,
    'aporp6': np.float64,
    'sbc_6': np.float64,
    'can_apor6': np.float64,
}
label_column_dtype = {"flag_cedido": np.float64}


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


if __name__ == "__main__":
    base_dir = "/opt/ml/processing"

    df = pd.read_csv(
        f"{base_dir}/input/desercion.csv",
        encoding="utf-8-sig",
        header=None,
        names=feature_columns_names + [label_column],
        dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype),
    )

    y = df.pop(label_column)
    X_pre = df

    df_batch = pd.DataFrame(X_pre)
    df_batch.to_csv(
        f"{base_dir}/batch/batch_features.csv",
        header=False,
        index=False,
        encoding="utf-8"
    )

    y_arr = y.to_numpy().reshape(-1, 1)
    full = np.hstack([y_arr, X_pre])
    np.random.shuffle(full)
    train, validation, test = np.split(
        full,
        [int(0.7 * len(full)), int(0.85 * len(full))]
    )

    pd.DataFrame(train).to_csv(
        f"{base_dir}/train/train.csv", header=False, index=False
    )
    pd.DataFrame(validation).to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )
    pd.DataFrame(test).to_csv(
        f"{base_dir}/test/test.csv", header=False, index=False
    )
