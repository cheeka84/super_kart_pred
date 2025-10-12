
# importing all important python libraries
import os
import numpy as np
import pandas as pd
import joblib
import mlflow

# importing scikit-learn (used for model training)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# importing hugging face hub libraries (used to upload model to hugging face)
from huggingface_hub import HfApi, create_repo
from huggingface_hub.errors import RepositoryNotFoundError



## configuration section##

# we will read the dataset splits on hugging face
SPLITS_REPO_ID = os.getenv("SPLITS_REPO_ID", "cheeka84/super-kart-pred")
SPLITS_SUBDIR  = os.getenv("SPLITS_SUBDIR", "splits")   # folder name in hf dataset

# getting hugging face token from environment 
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if not HF_TOKEN:
    raise SystemExit("hf_token not set. please set it before running.")

# connect mlflow to track model runs 
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "SuperKart-Sales-Regression"))

# making full hugging face dataset paths for reading csvs
prefix = f"hf://datasets/{SPLITS_REPO_ID}/"
base = f"{SPLITS_SUBDIR}/" if SPLITS_SUBDIR else ""
Xtrain_path = prefix + base + "Xtrain.csv"
Xtest_path  = prefix + base + "Xtest.csv"
ytrain_path = prefix + base + "ytrain.csv"
ytest_path  = prefix + base + "ytest.csv"



## load training and test data ##

# reading csv files from hugging face
Xtrain = pd.read_csv(Xtrain_path)
Xtest  = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest  = pd.read_csv(ytest_path)



## select features ##

# numeric columns in our dataset
numeric_features = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Establishment_Year",
]

# add extra engineered columns from prep.py
for eng in ["Store_Age", "Price_per_Area"]:
    if eng in Xtrain.columns:
        numeric_features.append(eng)
numeric_features = [c for c in numeric_features if c in Xtrain.columns]

# categorical columns (non-numeric)
categorical_features = [
    "Product_Sugar_Content",
    "Product_Type",
    "Store_Id",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
]
categorical_features = [c for c in categorical_features if c in Xtrain.columns]

# check if we have any valid features, else stop
if not numeric_features and not categorical_features:
    raise SystemExit("no valid features found in training data!")



## data preprocessing##

# this converts text columns into numbers using one hot encoding
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

# combine numeric and categorical processing together
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", make_ohe(), categorical_features),
    ]
)


## define models to train ##

# model 1: xgboost 
xgb_reg = xgb.XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42,
    tree_method="hist",
)

# model 2: random forest 
rf_reg = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

# set of values to test (for both models) to find best combination
xgb_grid = {
    "xgb__n_estimators": [200, 400, 500, 600],
    "xgb__max_depth": [5, 6, 8],
    "xgb__learning_rate": [0.005, 0.01, 0.05],
    "xgb__subsample": [0.8, 1.0],
    "xgb__colsample_bytree": [0.8, 1.0],
    "xgb__reg_lambda": [1.0, 2.0],
}

rf_grid = {
    "rf__n_estimators": [100, 200, 500],
    "rf__max_depth": [10, 12, 15],
    "rf__min_samples_split": [2, 5, 10],
    "rf__min_samples_leaf": [5, 8, 10],
    "rf__max_features": ["sqrt", "log2"],
}

# combine preprocessing and model together into a single pipeline
pipe_xgb = Pipeline([("prep", preprocessor), ("xgb", xgb_reg)])
pipe_rf  = Pipeline([("prep", preprocessor), ("rf", rf_reg)])



## metric function ##

# rmse = root mean square error (lower means better)
def rmse(y, yhat):
    return float(np.sqrt(mean_squared_error(y, yhat)))



## function to train + log model ##

def evaluate_and_log(prefix, model, Xtr, ytr, Xte, yte):
    """trains the model, checks scores, and logs results to mlflow"""
    grid = GridSearchCV(
        model,
        param_grid=xgb_grid if prefix == "xgb" else rf_grid,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(Xtr, ytr)
    best = grid.best_estimator_  # best model after grid search

    # predict on train and test
    yhat_tr = best.predict(Xtr)
    yhat_te = best.predict(Xte)

    # calculate different performance metrics
    metrics = {
        f"{prefix}_cv_neg_rmse": float(grid.best_score_),
        f"{prefix}_train_rmse": rmse(ytr, yhat_tr),
        f"{prefix}_train_mae": float(mean_absolute_error(ytr, yhat_tr)),
        f"{prefix}_train_r2": float(r2_score(ytr, yhat_tr)),
        f"{prefix}_test_rmse": rmse(yte, yhat_te),
        f"{prefix}_test_mae": float(mean_absolute_error(yte, yhat_te)),
        f"{prefix}_test_r2": float(r2_score(yte, yhat_te)),
    }

    # log best parameters and results to mlflow
    mlflow.log_params({f"{prefix}__" + k: v for k, v in grid.best_params_.items()})
    mlflow.log_metrics(metrics)

    # return final model info
    return {
        "best_estimator": best,
        "best_params": grid.best_params_,
        "metrics": metrics,
    }

## main training flow ##

with mlflow.start_run():
    # train xgboost
    res_xgb = evaluate_and_log("xgb", pipe_xgb, Xtrain, ytrain, Xtest, ytest)
    print("xgb best params:", res_xgb["best_params"])
    print("xgb metrics:", res_xgb["metrics"])

    # train random forest
    res_rf = evaluate_and_log("rf", pipe_rf, Xtrain, ytrain, Xtest, ytest)
    print("rf best params:", res_rf["best_params"])
    print("rf metrics:", res_rf["metrics"])

    # compare which model is better based on test rmse
    xgb_rmse = res_xgb["metrics"]["xgb_test_rmse"]
    rf_rmse  = res_rf["metrics"]["rf_test_rmse"]

    if xgb_rmse <= rf_rmse:
        chosen_name = "xgboost"
        chosen = res_xgb["best_estimator"]
        chosen_rmse = xgb_rmse
    else:
        chosen_name = "random_forest"
        chosen = res_rf["best_estimator"]
        chosen_rmse = rf_rmse

    # log the chosen model in mlflow
    mlflow.log_param("selected_model", chosen_name)
    mlflow.log_metric("selected_test_rmse", chosen_rmse)
    print(f"selected model: {chosen_name} (test rmse={chosen_rmse:.4f})")

    # save the selected model locally
    out_path = f"superkart_{chosen_name}_regressor.joblib"
    joblib.dump(chosen, out_path)
    mlflow.log_artifact(out_path, artifact_path="model")

   
    ## upload model to hugging face ##
   
    api = HfApi(token=HF_TOKEN)
    MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "cheeka84/super-kart-pred")
    try:
        api.repo_info(repo_id=MODEL_REPO_ID, repo_type="model")
        print(f"model repo exists: {MODEL_REPO_ID}")
    except RepositoryNotFoundError:
        print(f"creating model repo: {MODEL_REPO_ID}")
        create_repo(repo_id=MODEL_REPO_ID, repo_type="model", private=False, exist_ok=True, token=HF_TOKEN)

    # upload the best model file
    api.upload_file(
        path_or_fileobj=out_path,
        path_in_repo=os.path.basename(out_path),
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        commit_message=f"upload best regressor ({chosen_name}) for superkart sales prediction",
    )
    print(f"uploaded model to hf: {MODEL_REPO_ID}/{os.path.basename(out_path)}")
