import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,average_precision_score

def get_base_model(name:str,seed:int=42):

    name = name.lower().strip()

    if name in['logreg','logistic','logistic_regression']:
        return LogisticRegression(max_iter=2000)
    
    if name in ['rf','random_forest','randomforest']:
                return RandomForestClassifier(
                       n_estimators=300,
                       random_state=seed,
                       n_jobs=1,
                       min_samples_leaf=2
                )
    raise ValueError("Uknown model name")

def evaluate_binary_probs(y_true,p_pred):
       
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred)

    if p_pred.ndim !=1:
          raise ValueError("p_pred must be a 1D array of positive-class probabilities")
    if np.any(p_pred<0) or np.any(p_pred>1):
          raise ValueError('probablities must be in [0,1]')
    
    auc = roc_auc_score(y_true,p_pred)
    ap = average_precision_score(y_true,p_pred)

    return{"roc_auc":float(auc),"pr_auc":float(ap)}


def train_and_eval(model_name:str,
                   xt_train,y_train,
                   xt_val,y_val,
                   seed:int=42):
      
      model = get_base_model(model_name,seed=seed)
      model.fit(xt_train,y_train)

      p_val = model.predict_proba(xt_val)[:,1]
      metrics = evaluate_binary_probs(y_val,p_val)
      metrics['model']=model_name

      return model_name,metrics,p_val

if __name__=="__main__":
      
      from src.data.load_adult import load_adult
      from src.data.split import split_train_val_test
      from src.features.preprocess import infer_column_types,build_preprocessor,fit_transform_splits
      
      x,y,meta = load_adult(return_meta=True)
      x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(x, y)
      cat_cols,num_cols = infer_column_types(x_train)
      pre = build_preprocessor(cat_cols,num_cols,scale_numeric=True)
      xt_train,xt_val,xt_test = fit_transform_splits(pre,x_train,x_val,x_test)

      result =[]
      trained={}

      for model in ["logreg","rf"]:
            model,metrics,_=train_and_eval(
                  model,
                  xt_train,y_train,
                  xt_val,y_val,
                  seed=42
            )

            trained[model]=model
            result.append(metrics)
      print("\nValidation comparison:")
      print("Model      ROC-AUC    PR-AUC")
      print("----------------------------")
      for r in sorted(result, key=lambda d: d["pr_auc"], reverse=True):
            print(f"{r['model']:<10} {r['roc_auc']:<9.4f} {r['pr_auc']:<.4f}")
            