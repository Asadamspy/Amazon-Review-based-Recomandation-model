# app.py

import streamlit as st
import joblib
import pandas as pd

# --------------------------
# Load models and mappings
# --------------------------
@st.cache_resource
def load_model_files():
    svd_model = joblib.load("svd_amazon_model.joblib")
    item_map = joblib.load("item_id_map.joblib")
    user_map = joblib.load("user_id_map.joblib")
    return svd_model, item_map, user_map

svd_model, item_map, user_map = load_model_files()

# Reverse maps for display
inv_item_map = {v: k for k, v in item_map.items()}

# --------------------------
# Streamlit App Layout
# --------------------------
st.set_page_config(page_title="Amazon Product Recommender", layout="wide")
st.title("🛒 Amazon Product Recommendation Engine")
st.markdown(
    "Select a user ID to get Top-5 recommended products based on past reviews."
)

# User selection
user_id_input = st.selectbox(
    "Select User ID",
    options=list(user_map.values())
)

# --------------------------
# Recommendation Function
# --------------------------
def recommend_products(user_id, top_n=5):
    """Return top-N product recommendations for a given user ID."""
    try:
        uid_enc = [k for k, v in user_map.items() if v == user_id][0]
    except IndexError:
        return []

    # Build anti-testset for this user
    user_items = [iid for iid in item_map.values()]
    predictions = [
        (iid, svd_model.predict(uid_enc, iid).est) for iid in user_items
    ]

    # Sort and take top-N
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_items = [inv_item_map[iid] for iid, _ in predictions[:top_n]]
    return top_items

# --------------------------
# Show Recommendations
# --------------------------
if st.button("Get Recommendations"):
    with st.spinner("Calculating recommendations..."):
        top_products = recommend_products(user_id_input, top_n=5)
        if top_products:
            st.success(f"Top-5 recommended products for User {user_id_input}:")
            for idx, prod in enumerate(top_products, 1):
                st.write(f"{idx}. {prod}")
        else:
            st.warning("No recommendations found for this user.")

