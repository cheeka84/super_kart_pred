
import os
os.environ.setdefault("HOME", "/tmp")
os.makedirs(os.path.expanduser("~/.streamlit"), exist_ok=True)

# importing required libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from huggingface_hub import hf_hub_download

# set app title and layout
st.set_page_config(page_title="superkart sales prediction")

st.title("superkart — predict product-store sales (regression)")
st.caption("enter product & store details to predict `product_store_sales_total`")

## model download/load ##

# model repo name where the trained model is uploaded
MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "cheeka84/super-kart-pred")

# we don’t know which model won (xgboost or random forest), so try both
CANDIDATE_FILES = [
    "superkart_xgboost_regressor.joblib",
    "superkart_random_forest_regressor.joblib",
]

# get hugging face token if needed for private repo
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

# function to download and load the model file
def load_model():
    last_err = None
    for fname in CANDIDATE_FILES:
        try:
            path = hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename=fname,
                repo_type="model",
                token=HF_TOKEN  # remove if repo is public
            )
            return joblib.load(path), fname
        except Exception as e:
            last_err = e
    raise RuntimeError(f"could not download any model from {MODEL_REPO_ID}. "
                       f"tried: {CANDIDATE_FILES}. last error: {last_err}")

# load the trained model
model, model_file = load_model()
st.success(f"loaded model: `{model_file}` from {MODEL_REPO_ID}")


## input ui (same as training features) ##

# divide form into two columns
col1, col2 = st.columns(2)
with col1:
    product_weight = st.number_input("product_weight", min_value=0.0, value=10.0, step=1.0)
    product_area   = st.number_input("product_allocated_area", min_value=0.0, value=0.01, step=0.01)
    product_mrp    = st.number_input("product_mrp", min_value=0.0, value=10.0, step=1.0)
    est_year       = st.number_input("store_establishment_year", min_value=1950, max_value=datetime.now().year, value=2015, step=1)

with col2:
    sugar_content  = st.selectbox("product_sugar_content", options=["low sugar", "regular", "no sugar"])
    product_type   = st.selectbox("product_type", options=["frozen foods","dairy","canned","baking goods","health and hygiene","snack foods","meat","household",
    "hard drinks","fruits and vegetables","breads","soft drinks","breakfast","others","starchy foods","seafood"])
    store_id       = st.selectbox("store_id", options=["OUT001", "OUT002", "OUT003", "OUT004"])
    store_size     = st.selectbox("store_size", options=["small", "medium", "high"])
    city_type      = st.selectbox("store_location_city_type", options=["tier 1", "tier 2", "tier 3"])
    store_type     = st.selectbox("store_type", options=['supermarket type2', 'departmental store', 'supermarket type1', 'food mart'])

# engineered features (same as in prep.py)
current_year = datetime.now().year
store_age     = max(0, min(200, current_year - int(est_year)))  # limit age between 0 and 200
price_per_area = float(product_mrp) / float(product_area) if product_area not in (0, None) else 0.0

# make one row dataframe with all input values
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

# show the input data entered by user
st.subheader("input preview")
st.dataframe(input_df)


## prediction ##

if st.button("predict sales"):
    try:
        # model expects dataframe with same columns used while training
        pred = model.predict(input_df)[0]
        st.markdown(f"### 🔮 predicted `product_store_sales_total`: **{pred:,.2f}**")
    except Exception as e:
        st.error(f"prediction failed: {e}")
        st.exception(e)

# small note for user
st.info("note: unknown category values are safely ignored by one-hot encoder.")

