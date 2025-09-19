import streamlit as st
import joblib
from collections import defaultdict

# --- Load model and mappings ---
model = joblib.load('svd_amazon_model.joblib')
item_map = joblib.load('item_id_map.joblib')
user_map = joblib.load('user_id_map.joblib')

# Function to get Top-N recommendations
def get_top_n(user_id_enc, n=5):
    all_items = list(item_map.keys())
    predictions = []
    
    for iid in all_items:
        pred = model.predict(user_id_enc, iid)
        predictions.append((iid, pred.est))
    
    # Sort and take top-N
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n = [item_map[iid] for iid, _ in predictions[:n]]
    return top_n

# --- Streamlit UI ---
st.title("Amazon Product Recommendation Engine")
st.write("Enter a User ID to get Top-5 recommended products.")

user_input = st.text_input("UserId")

if st.button("Get Recommendations"):
    # Find encoded user id
    try:
        user_id_enc = [k for k, v in user_map.items() if v == user_input][0]
        recommendations = get_top_n(user_id_enc, n=5)
        st.success("Top-5 Recommended Products:")
        for i, prod in enumerate(recommendations, 1):
            st.write(f"{i}. {prod}")
    except IndexError:
        st.error("UserId not found in the dataset!")
