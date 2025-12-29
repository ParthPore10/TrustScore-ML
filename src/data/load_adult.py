import pandas as pd
import numpy as np

from sklearn.datasets import fetch_openml

def load_adult(return_meta:bool=True):
    adult = fetch_openml(name='adult',version=2,as_frame=True)
    df = adult.frame.copy()

    if "class" not in df.columns:
        raise ValueError("target column not found")

    y_raw = df['class']
    x = df.drop(columns = ['class'])

    y = y_raw.str.strip().map({">50K":1, "<=50K":0})

    if y.isnull().any():
        raise ValueError("Target mappings produced NANs")
    
    obj_cols = x.select_dtypes(include=['object']).columns
    for cols in obj_cols:
        x[cols] = x[cols].str.strip()
    
    x.replace("?",np.nan,inplace=True)

    assert 'class' not in x.columns
    assert len(x)==len(y)

    if not(return_meta):
        return x,y
    
    categorical_cols = x.select_dtypes(include=['object']).columns
    numerical_cols = x.select_dtypes(exclude = ['object']).columns

    meta={
        "source":"openml",
        "dataset":'adult',
        "version":2,
        "n_rows":x.shape[0],
        "n_cols":x.shape[1],
        "categorical_cols":categorical_cols,
        "numerical_cols":numerical_cols,
        "missing_per_col":x.isna().sum().sort_values(ascending=False).to_dict(),
        "target_positive_rate":float(y.mean())
    }

    return x,y,meta

if __name__=="__main__":
    x,y,meta = load_adult()
    
