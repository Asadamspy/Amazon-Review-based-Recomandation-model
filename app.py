import streamlit as st
import pandas as pd
import numpy as np
import joblib
from surprise import SVD

# --- Load Model and Mappings ---
@st.cache_data
def load_model():
    model = joblib.load('svd_amazon_model.joblib')
    item_map = joblib.load('item_id_map.joblib')
    user_map = joblib.load('user_id_map.joblib')
    return model, item_map, user_map

model, item_map, user_map = load_model()

# Reverse mapping for displaying ProductIds
rev_item_map = {v: k for k, v in item_map.items()}

# --- Streamlit UI ---
st.set_page_config(page_title="Amazon Product Recommender", layout="wide")
st.title("Amazon Product Recommendation System")
st.markdown("Get Top-N product recommendations for any user!")

# Select user
user_ids = list(user_map.values())
selected_user = st.selectbox("Select User ID:", user_ids)

if st.button("Recommend Products"):
    # Encode user ID
    uid_enc = [k for k, v in user_map.items() if v == selected_user][0]

    # Build anti-testset for this user
    all_items = list(item_map.values())
    user_rated_items = [iid for (iid, _) in model.trainset.ur[uid_enc]] if uid_enc in model.trainset.ur else []
    anti_testset = [(uid_enc, iid, 0) for iid in all_items if iid not in user_rated_items]

    # Predict
    predictions = model.test(anti_testset)

    # Get top 5
    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:5]
    recommended_products = [rev_item_map[int(iid)] for (uid, iid, true_r, est) in top_n]

    st.success(f"Top 5 recommended Product IDs for User {selected_user}:")
    st.write(recommended_products)
