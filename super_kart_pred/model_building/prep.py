# ------------------------------------------------------------
# SuperKart Regression Data Preparation Script
# ------------------------------------------------------------
# This script automates the MLOps data-preparation stage:
#   1. Load raw data (local or from Hugging Face Hub)
#   2. Clean & normalize values
#   3. Perform basic feature engineering
#   4. Split into train/test datasets
#   5. Upload processed splits back to Hugging Face Hub
#
# Purpose: Ensures fully reproducible, version-controlled datasets
#           for downstream model training pipelines.
# ------------------------------------------------------------

import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, create_repo
from huggingface_hub.errors import RepositoryNotFoundError

# ----------------------------
# CONFIGURATION
# ----------------------------
# Define where the raw dataset resides. Prefer local file if available,
# otherwise fall back to Hugging Face remote dataset URI.
LOCAL_DATASET_PATH = "/mnt/data/SuperKart_mlops.csv"
HF_DATASET_PATH = "hf://datasets/cheeka84/super-kart-pred/SuperKart_mlops.csv"

# Destination Hugging Face dataset repository for storing processed splits.
REPO_ID = "cheeka84/super-kart-pred"      # change if you want to store elsewhere
REPO_TYPE = "dataset"

# Retrieve your Hugging Face authentication token from environment.
# This token is mandatory for upload access (use CI secrets or local env vars).
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if not HF_TOKEN:
    raise SystemExit("HF_TOKEN/HUGGINGFACE_HUB_TOKEN not set. Export it or pass via CI env.")

# Initialize Hugging Face API client.
api = HfApi(token=HF_TOKEN)

# ----------------------------
# VALIDATE OR CREATE DATASET REPO
# ----------------------------
try:
    # Try fetching repo info; if found, it already exists.
    api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"Dataset repo exists: {REPO_ID}")
except RepositoryNotFoundError:
    # If repository not found, create a new public dataset repo.
    print(f"Creating dataset repo: {REPO_ID}")
    create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        private=False,
        exist_ok=True,
        token=HF_TOKEN,
    )

# ----------------------------
# LOAD DATASET
# ----------------------------
# Prefer the local CSV if available, else fallback to Hugging Face storage.
src_path = LOCAL_DATASET_PATH if os.path.isfile(LOCAL_DATASET_PATH) else HF_DATASET_PATH
print("Reading dataset from:", src_path)

# Attempt to read CSV; provide descriptive error if hf:// protocol fails.
try:
    superkart_df = pd.read_csv(src_path)
except Exception as e:
    raise SystemExit(
        f"Failed to read dataset from {src_path}. "
        f"If using hf:// make sure 'huggingface_hub[fsspec]' is installed. Error: {e}"
    )

print("Dataset loaded successfully. Shape:", superkart_df.shape)

# ----------------------------
# DEFINE TARGET AND FEATURES
# ----------------------------
# The column we want to predict:
target = "Product_Store_Sales_Total"

# Base numeric features expected in the raw dataset
numeric_features = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Establishment_Year",
]

# Base categorical features
categorical_features = [
    "Product_Id",
    "Product_Sugar_Content",
    "Product_Type",
    "Store_Id",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
]

# ----------------------------
# DATA CLEANING
# ----------------------------
cols_to_drop = []

# Drop almost-unique ID columns that do not add predictive power
if "Product_Id" in superkart_df.columns:
    cols_to_drop.append("Product_Id")

# Remove duplicate rows to ensure clean training data
before = len(superkart_df)
superkart_df = superkart_df.drop_duplicates()
print("Dropped duplicates:", before - len(superkart_df))

# Normalize inconsistent text labels — fix 'reg' → 'Regular'
superkart_df['Product_Sugar_Content'] = superkart_df['Product_Sugar_Content'].replace(['reg'], "Regular")

# Trim and normalize whitespace in all string/object columns
for col in superkart_df.select_dtypes(include="object").columns:
    superkart_df[col] = (
        superkart_df[col]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

# Convert numeric columns from strings to numbers, coercing invalid entries to NaN
for col in ["Product_Weight", "Product_Allocated_Area", "Product_MRP",
            "Store_Establishment_Year", "Product_Store_Sales_Total"]:
    if col in superkart_df.columns:
        superkart_df[col] = pd.to_numeric(superkart_df[col], errors="coerce")

# Drop any rows missing the target variable — essential for supervised learning
before = len(superkart_df)
superkart_df = superkart_df.dropna(subset=[target])
print("Dropped rows with missing target:", before - len(superkart_df))

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
current_year = datetime.now().year

# Derive "Store_Age" feature from establishment year
if "Store_Establishment_Year" in superkart_df.columns:
    superkart_df["Store_Age"] = (
        current_year - superkart_df["Store_Establishment_Year"]
    ).clip(lower=0, upper=200)  # avoid negative/absurd ages

# Compute "Price_per_Area" safely (handle zero divisions)
if "Product_MRP" in superkart_df.columns and "Product_Allocated_Area" in superkart_df.columns:
    denom = superkart_df["Product_Allocated_Area"].replace(0, np.nan)
    superkart_df["Price_per_Area"] = superkart_df["Product_MRP"] / denom
    superkart_df["Price_per_Area"] = superkart_df["Price_per_Area"].replace([np.inf, -np.inf], np.nan)
    superkart_df["Price_per_Area"] = superkart_df["Price_per_Area"].fillna(
        superkart_df["Price_per_Area"].median()
    )

# Compile the final feature lists after engineering
final_numeric = [c for c in numeric_features if c in superkart_df.columns]
if "Store_Age" in superkart_df.columns:
    final_numeric.append("Store_Age")
if "Price_per_Area" in superkart_df.columns:
    final_numeric.append("Price_per_Area")

final_categorical = [
    c for c in categorical_features if c in superkart_df.columns and c not in cols_to_drop
]

# Validate all required columns exist
missing = [c for c in (final_numeric + final_categorical + [target]) if c not in superkart_df.columns]
if missing:
    raise SystemExit(f"Missing expected columns after cleaning/engineering: {missing}")

# Separate features and target for training
X = superkart_df[final_numeric + final_categorical].copy()
y = superkart_df[target].copy()

print("Final shapes — X:", X.shape, " y:", y.shape)
print("Numeric features:", final_numeric)
print("Categorical features:", final_categorical)

# ----------------------------
# TRAIN/TEST SPLIT
# ----------------------------
# Perform an 80/20 split for model training and validation
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Save split files locally (CSV format for portability)
for fname, obj in [("Xtrain.csv", Xtrain), ("Xtest.csv", Xtest), ("ytrain.csv", ytrain), ("ytest.csv", ytest)]:
    Path(fname).write_text("") if False else None  # placeholder (no-op, prevents linter warnings)
    obj.to_csv(fname, index=False)
    print("Saved", Path(fname).resolve())

# ----------------------------
# UPLOAD TO HUGGING FACE HUB
# ----------------------------
# Upload processed splits to the dataset repository under a subfolder "splits/"
# This keeps data versions organized and supports lineage tracking for MLflow.
for file_path in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    print(f"Uploading {file_path} -> {REPO_ID}")
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=f"splits/{os.path.basename(file_path)}",  # maintain clear folder structure
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="Add regression splits with engineered features",
    )

print("Upload complete.")
