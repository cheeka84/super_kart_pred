import os
# --- make Streamlit writable in containers (avoids '/.streamlit' PermissionError)
os.environ.setdefault("HOME", "/tmp")
os.makedirs(os.path.expanduser("~/.streamlit"), exist_ok=True)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="SuperKart Sales Prediction", page_icon="🛒")

st.title("🛒 SuperKart — Predict Product-Store Sales (Regression)")
st.caption("Enter product & store attributes to predict `Product_Store_Sales_Total`")

# ----------------------------
# Model download/load
# ----------------------------
# Set your model repo (where train.py uploaded the chosen regressor)
MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "cheeka84/super-kart-pred")
# We don't know which won (XGBoost/RandomForest), so try both filenames:
CANDIDATE_FILES = [
    "superkart_xgboost_regressor.joblib",
    "superkart_random_forest_regressor.joblib",
]
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

def load_model():
    last_err = None
    for fname in CANDIDATE_FILES:
        try:
            path = hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename=fname,
                repo_type="model",
                token=HF_TOKEN  # omit if repo is public
            )
            return joblib.load(path), fname
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not download any model from {MODEL_REPO_ID}. "
                       f"Tried: {CANDIDATE_FILES}. Last error: {last_err}")

model, model_file = load_model()
st.success(f"Loaded model: `{model_file}` from {MODEL_REPO_ID}")

# ----------------------------
# Input UI (match training features)
# ----------------------------
col1, col2 = st.columns(2)
with col1:
    product_weight = st.number_input("Product_Weight", min_value=0.0, value=10.0, step=1.0)
    product_area   = st.number_input("Product_Allocated_Area", min_value=0.0, value=0.01, step=0.01)
    product_mrp    = st.number_input("Product_MRP", min_value=0.0, value=10.0, step=1.0)
    est_year       = st.number_input("Store_Establishment_Year", min_value=1950, max_value=datetime.now().year, value=2015, step=1)

with col2:
    sugar_content  = st.selectbox("Product_Sugar_Content", options=["Low Sugar", "Regular", "No Sugar"])
    product_type   = st.selectbox("Product_Type", options=["Frozen Foods","Dairy","Canned","Baking Goods","Health and Hygiene","Snack Foods","Meat","Household",
    "Hard Drinks","Fruits and Vegetables","Breads","Soft Drinks","Breakfast","Others","Starchy Foods","Seafood"])
    store_id       = st.selectbox("Store_Id", options=["OUT001", "OUT002", "OUT003", "OUT004"])
    store_size     = st.selectbox("Store_Size", options=["Small", "Medium", "High"])
    city_type      = st.selectbox("Store_Location_City_Type", options=["Tier 1", "Tier 2", "Tier 3"])
    store_type     = st.selectbox("Store_Type", options=['Supermarket Type2', 'Departmental Store', 'Supermarket Type1', 'Food Mart'])

# Engineered features (prep.py added these; compute here as well)
current_year = datetime.now().year
store_age     = max(0, min(200, current_year - int(est_year)))  # clip [0,200]
price_per_area = float(product_mrp) / float(product_area) if product_area not in (0, None) else 0.0

# Build single-row DataFrame with ALL expected columns (extras are fine)
row = {
    "Product_Weight": product_weight,
    "Product_Allocated_Area": product_area,
    "Product_MRP": product_mrp,
    "Store_Establishment_Year": est_year,
    "Store_Age": store_age,
    "Price_per_Area": price_per_area,
    "Product_Sugar_Content": sugar_content.strip(),
    "Product_Type": product_type.strip(),
    "Store_Id": store_id.strip(),
    "Store_Size": store_size.strip(),
    "Store_Location_City_Type": city_type.strip(),
    "Store_Type": store_type.strip(),
}
input_df = pd.DataFrame([row])

st.subheader("Input preview")
st.dataframe(input_df)

# ----------------------------
# Predict
# ----------------------------
if st.button("Predict sales"):
    try:
        # Pipeline expects DataFrame with training column names; we provided them.
        pred = model.predict(input_df)[0]
        st.markdown(f"### 🔮 Predicted `Product_Store_Sales_Total`: **{pred:,.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)

st.info("Note: Unknown category values are safely ignored by the one-hot encoder (handled as all-zero columns).")
