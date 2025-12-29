import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer

def infer_column_types(x_train:pd.DataFrame):
    
    if not isinstance(x_train,pd.DataFrame):
        raise TypeError("x_train nust be a pandas dataframe")
    
    cat_cols = x_train.select_dtypes(include=['object','category']).columns.tolist()
    num_cols = [i for i in x_train.columns if i not in cat_cols]

    if len(set(cat_cols).intersection(set(num_cols)))!=0:
        raise ValueError("column overlap detected between numerical and categorical columns")
    if len(cat_cols)+len(num_cols)!=x_train.shape[1]:
        raise ValueError("column mismatch")
    
    return cat_cols,num_cols

def build_preprocessor(cat_cols,num_cols,scale_numeric:bool=True)->ColumnTransformer:

    numeric_steps = [("imputer",SimpleImputer(strategy="median"))]

    if scale_numeric:
        numeric_steps.append(('scaler',StandardScaler(with_mean=False)))
    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(steps=[
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num",numeric_pipeline,num_cols),
            ("cat",categorical_pipeline,cat_cols)
        ],
        remainder= "drop"
    )

    return preprocessor

def fit_transform_splits(preprocessor:ColumnTransformer,
                         x_train:pd.DataFrame,
                         x_val:pd.DataFrame,
                         x_test:pd.DataFrame):
    
    preprocessor.fit(x_train)

    xt_train = preprocessor.transform(x_train)
    xt_val = preprocessor.transform(x_val)
    xt_test = preprocessor.transform(x_test)

    n_features = xt_train.shape[1]

    if xt_val.shape[1]!=n_features or xt_test.shape[1]!=n_features:
        raise ValueError("number of columns are mismatched")
    
    def _has_nan(m)->bool:
        if hasattr(m,"data"):
            return np.isnan(m.data).any()
        return np.isnan(np.asarray(m)).any()
    
    if _has_nan(xt_train) or _has_nan(xt_val) or _has_nan(xt_train):
        raise ValueError(" NAN found after preprocessing")
    
    return xt_train,xt_val,xt_test

if __name__ =='__main__':
    from src.data.load_adult import load_adult
    from src.data.split import split_train_val_test

    x,y,meta = load_adult(return_meta=True)
    x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(x, y)

    cat_cols,num_cols = infer_column_types(x_train)
    pre = build_preprocessor(cat_cols,num_cols,scale_numeric=True)
    xt_train,xt_val,xt_test = fit_transform_splits(pre,x_train,x_val,x_test)