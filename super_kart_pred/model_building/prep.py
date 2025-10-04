# SuperKart regression: load -> clean -> feature engineer -> split -> upload

import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, create_repo
from huggingface_hub.errors import RepositoryNotFoundError

# ----------------------------
# Config
# ----------------------------
LOCAL_DATASET_PATH = "/mnt/data/SuperKart_mlops.csv"
HF_DATASET_PATH = "hf://datasets/cheeka84/super-kart-pred/SuperKart_mlops.csv"

# Use a separate repo for splits (recommended), or keep the same:
REPO_ID = "cheeka84/super-kart-pred"   # change to your preferred dataset repo
REPO_TYPE = "dataset"

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if not HF_TOKEN:
    raise SystemExit("HF_TOKEN/HUGGINGFACE_HUB_TOKEN not set. Export it or pass via CI env.")

api = HfApi(token=HF_TOKEN)

# Ensure destination dataset repo exists
try:
    api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"Dataset repo exists: {REPO_ID}")
except RepositoryNotFoundError:
    print(f"Creating dataset repo: {REPO_ID}")
    create_repo(
        repo_id=REPO_ID, repo_type=REPO_TYPE, private=False, exist_ok=True, token=HF_TOKEN
    )

# ----------------------------
# Load dataset
# ----------------------------
src_path = LOCAL_DATASET_PATH if os.path.isfile(LOCAL_DATASET_PATH) else HF_DATASET_PATH
print("Reading dataset from:", src_path)
try:
    superkart_df = pd.read_csv(src_path)
except Exception as e:
    raise SystemExit(
        f"Failed to read dataset from {src_path}. "
        f"If using hf:// make sure 'huggingface_hub[fsspec]' is installed. Error: {e}"
    )
print("Dataset loaded successfully. Shape:", superkart_df.shape)

# ----------------------------
# Target & base features (actual columns in this CSV)
# ----------------------------
target = "Product_Store_Sales_Total"

numeric_features = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Establishment_Year",
]

categorical_features = [
    "Product_Id",                 # will drop (too unique)
    "Product_Sugar_Content",
    "Product_Type",
    "Store_Id",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
]

# ----------------------------
# Cleaning
# ----------------------------
# Drop nearly-unique id
cols_to_drop = []
if "Product_Id" in superkart_df.columns:
    cols_to_drop.append("Product_Id")

# Deduplicate
before = len(superkart_df)
superkart_df = superkart_df.drop_duplicates()
print("Dropped duplicates:", before - len(superkart_df))

# Trim string columns & normalize spacing
for col in superkart_df.select_dtypes(include="object").columns:
    superkart_df[col] = superkart_df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

# Coerce numeric types
for col in ["Product_Weight", "Product_Allocated_Area", "Product_MRP",
            "Store_Establishment_Year", "Product_Store_Sales_Total"]:
    if col in superkart_df.columns:
        superkart_df[col] = pd.to_numeric(superkart_df[col], errors="coerce")

# Drop rows with missing target
before = len(superkart_df)
superkart_df = superkart_df.dropna(subset=[target])
print("Dropped rows with missing target:", before - len(superkart_df))

# Actually drop the id column
if cols_to_drop:
    superkart_df = superkart_df.drop(columns=cols_to_drop, errors="ignore")
    print("Dropped columns:", cols_to_drop)

# ----------------------------
# Feature engineering
# ----------------------------
current_year = datetime.now().year

# Store_Age
if "Store_Establishment_Year" in superkart_df.columns:
    superkart_df["Store_Age"] = (current_year - superkart_df["Store_Establishment_Year"]).clip(lower=0, upper=200)

# Price_per_Area (safe division)
if "Product_MRP" in superkart_df.columns and "Product_Allocated_Area" in superkart_df.columns:
    denom = superkart_df["Product_Allocated_Area"].replace(0, np.nan)
    superkart_df["Price_per_Area"] = superkart_df["Product_MRP"] / denom
    superkart_df["Price_per_Area"] = superkart_df["Price_per_Area"].replace([np.inf, -np.inf], np.nan)
    superkart_df["Price_per_Area"] = superkart_df["Price_per_Area"].fillna(superkart_df["Price_per_Area"].median())

# Final feature lists
final_numeric = [c for c in numeric_features if c in superkart_df.columns]
if "Store_Age" in superkart_df.columns:
    final_numeric.append("Store_Age")
if "Price_per_Area" in superkart_df.columns:
    final_numeric.append("Price_per_Area")

final_categorical = [
    c for c in categorical_features if c in superkart_df.columns and c not in cols_to_drop
]

missing = [c for c in (final_numeric + final_categorical + [target]) if c not in superkart_df.columns]
if missing:
    raise SystemExit(f"Missing expected columns after cleaning/engineering: {missing}")

X = superkart_df[final_numeric + final_categorical].copy()
y = superkart_df[target].copy()

print("Final shapes — X:", X.shape, " y:", y.shape)
print("Numeric features:", final_numeric)
print("Categorical features:", final_categorical)

# ----------------------------
# Train/test split
# ----------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Save locally
for fname, obj in [("Xtrain.csv", Xtrain), ("Xtest.csv", Xtest), ("ytrain.csv", ytrain), ("ytest.csv", ytest)]:
    Path(fname).write_text("") if False else None  # placeholder to avoid lints
    obj.to_csv(fname, index=False)
    print("Saved", Path(fname).resolve())

# ----------------------------
# Upload to Hugging Face (you can put them under a folder like 'splits/')
# ----------------------------
for file_path in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    print(f"Uploading {file_path} -> {REPO_ID}")
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=f"splits/{os.path.basename(file_path)}",  # nicer structure
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="Add regression splits with engineered features",
    )
print("Upload complete.")
