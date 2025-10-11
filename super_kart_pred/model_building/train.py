import os
import numpy as np
import pandas as pd
import joblib
import mlflow

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from huggingface_hub import HfApi, create_repo
from huggingface_hub.errors import RepositoryNotFoundError

# =========================
# CONFIG — adjust if needed
# =========================
SPLITS_REPO_ID = os.getenv("SPLITS_REPO_ID", "cheeka84/super-kart-pred")  # repo with CSV splits
SPLITS_SUBDIR  = os.getenv("SPLITS_SUBDIR", "splits")                      # or "" if at root

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if not HF_TOKEN:
    raise SystemExit("HF_TOKEN/HUGGINGFACE_HUB_TOKEN not set. Export or pass via CI.")

# MLflow: use local file store if server not provided
mlflow.set_tracking_uri("http://localhost:5000")
# mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "SuperKart-Sales-Regression"))

# Paths to splits on HF Hub
prefix = f"hf://datasets/{SPLITS_REPO_ID}/"
base = f"{SPLITS_SUBDIR}/" if SPLITS_SUBDIR else ""
Xtrain_path = prefix + base + "Xtrain.csv"
Xtest_path  = prefix + base + "Xtest.csv"
ytrain_path = prefix + base + "ytrain.csv"
ytest_path  = prefix + base + "ytest.csv"

# ===========
# Load splits
# ===========
Xtrain = pd.read_csv(Xtrain_path)
Xtest  = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()  # Series
ytest  = pd.read_csv(ytest_path).squeeze()

# =========================
# Feature lists (SuperKart)
# =========================
numeric_features = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Establishment_Year",
]
# engineered features from prep.py if present
for eng in ["Store_Age", "Price_per_Area"]:
    if eng in Xtrain.columns:
        numeric_features.append(eng)
numeric_features = [c for c in numeric_features if c in Xtrain.columns]

categorical_features = [
    "Product_Sugar_Content",
    "Product_Type",
    "Store_Id",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
]
categorical_features = [c for c in categorical_features if c in Xtrain.columns]

if not numeric_features and not categorical_features:
    raise SystemExit("No usable features found in Xtrain. Check your splits/features.")

# OneHotEncoder compatibility across sklearn versions
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", make_ohe(), categorical_features),
    ]
)

# =========================
# Define models & grids
# =========================
xgb_reg = xgb.XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42,
    tree_method="hist",
)
rf_reg = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

xgb_grid = {
    "xgb__n_estimators": [200, 400, 500, 600],
    "xgb__max_depth": [5,6,8],
    "xgb__learning_rate": [0.005,0.01,0.05],
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

pipe_xgb = Pipeline([("prep", preprocessor), ("xgb", xgb_reg)])
pipe_rf  = Pipeline([("prep", preprocessor), ("rf", rf_reg)])

def rmse(y, yhat):
    return float(np.sqrt(mean_squared_error(y, yhat)))

def evaluate_and_log(prefix, model, Xtr, ytr, Xte, yte):
    """Fit model, compute metrics, and log to MLflow (returns dict)."""
    grid = GridSearchCV(
        model,
        param_grid=xgb_grid if prefix == "xgb" else rf_grid,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(Xtr, ytr)
    best = grid.best_estimator_

    yhat_tr = best.predict(Xtr)
    yhat_te = best.predict(Xte)

    metrics = {
        f"{prefix}_cv_neg_rmse": float(grid.best_score_),
        f"{prefix}_train_rmse": rmse(ytr, yhat_tr),
        f"{prefix}_train_mae": float(mean_absolute_error(ytr, yhat_tr)),
        f"{prefix}_train_r2": float(r2_score(ytr, yhat_tr)),
        f"{prefix}_test_rmse": rmse(yte, yhat_te),
        f"{prefix}_test_mae": float(mean_absolute_error(yte, yhat_te)),
        f"{prefix}_test_r2": float(r2_score(yte, yhat_te)),
    }
    # Log params & metrics
    mlflow.log_params({f"{prefix}__" + k: v for k, v in grid.best_params_.items()})
    mlflow.log_metrics(metrics)

    return {
        "best_estimator": best,
        "best_params": grid.best_params_,
        "metrics": metrics,
    }

with mlflow.start_run():
    # XGB
    res_xgb = evaluate_and_log("xgb", pipe_xgb, Xtrain, ytrain, Xtest, ytest)
    print("XGB best params:", res_xgb["best_params"])
    print("XGB metrics:", res_xgb["metrics"])

    # RF
    res_rf = evaluate_and_log("rf", pipe_rf, Xtrain, ytrain, Xtest, ytest)
    print("RF best params:", res_rf["best_params"])
    print("RF metrics:", res_rf["metrics"])

    # =========================
    # Choose the BEST by test RMSE
    # =========================
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

    mlflow.log_param("selected_model", chosen_name)
    mlflow.log_metric("selected_test_rmse", chosen_rmse)
    print(f"Selected model: {chosen_name} (test RMSE={chosen_rmse:.4f})")

    # Save chosen model
    out_path = f"superkart_{chosen_name}_regressor.joblib"
    joblib.dump(chosen, out_path)
    mlflow.log_artifact(out_path, artifact_path="model")

    # =========================
    # Push chosen model to HF Hub
    # =========================
    api = HfApi(token=HF_TOKEN)
    MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "cheeka84/super-kart-pred")
    try:
        api.repo_info(repo_id=MODEL_REPO_ID, repo_type="model")
        print(f"Model repo exists: {MODEL_REPO_ID}")
    except RepositoryNotFoundError:
        print(f"Creating model repo: {MODEL_REPO_ID}")
        create_repo(repo_id=MODEL_REPO_ID, repo_type="model", private=False, exist_ok=True, token=HF_TOKEN)

    api.upload_file(
        path_or_fileobj=out_path,
        path_in_repo=os.path.basename(out_path),
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        commit_message=f"Upload best regressor ({chosen_name}) for SuperKart sales prediction",
    )
    print(f"Uploaded model to HF: {MODEL_REPO_ID}/{os.path.basename(out_path)}")
