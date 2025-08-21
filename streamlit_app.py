import streamlit as st
import requests


API_URL = "http://localhost:8000"

st.title("Federated ZKP Credit Card Fraud Detection Simulator")

# Login as Bank 1 or Bank 2
bank = st.sidebar.selectbox("Login as", ["Bank 1", "Bank 2"])

# Get transaction count
count_resp = requests.get(f"{API_URL}/transactions/count")
if count_resp.status_code == 200:
    count = count_resp.json()["count"]
    sample_id = st.number_input("Select Transaction ID", min_value=0, max_value=count-1, value=0)
else:
    st.error("Could not fetch transaction count from API.")
    st.stop()

st.header("Federated Learning")
st.header("ZKP Inference and Verification")

st.header(f"{bank} Actions")

if bank == "Bank 1":
    if st.button("Train Bank 1 Model"):
        resp = requests.post(f"{API_URL}/federated/train", json={"bank": "bank1"})
        if resp.status_code == 200:
            st.success("Bank 1 model trained and saved.")
        else:
            st.error(f"Error: {resp.text}")
    if st.button("Send Transaction as Bank 1"):
        resp = requests.post(f"{API_URL}/bank1/classify", json={"sample_id": sample_id})
        if resp.status_code == 200:
            st.success(f"Bank 1 sent transaction #{sample_id}.")
            st.json(resp.json())
        else:
            st.error(f"Error: {resp.text}")
    if st.button("Receive and Validate Transaction as Bank 1"):
        resp = requests.get(f"{API_URL}/bank2/verify")
        if resp.status_code == 200:
            result = resp.json()
            st.success(f"Bank 1 received and verified transaction #{result['sample_id']}.")
            st.write(f"Prediction: {'FRAUD' if result['prediction'] else 'NOT FRAUD'}")
            st.write(f"ZKP Verified: {result['verified']}")
            if result.get('true_label') is not None:
                st.write(f"True Label: {result['true_label']}" )
        else:
            st.error(f"Error: {resp.text}")
elif bank == "Bank 2":
    if st.button("Train Bank 2 Model"):
        resp = requests.post(f"{API_URL}/federated/train", json={"bank": "bank2"})
        if resp.status_code == 200:
            st.success("Bank 2 model trained and saved.")
        else:
            st.error(f"Error: {resp.text}")
    if st.button("Send Transaction as Bank 2"):
        resp = requests.post(f"{API_URL}/bank1/classify", json={"sample_id": sample_id})
        if resp.status_code == 200:
            st.success(f"Bank 2 sent transaction #{sample_id}.")
            st.json(resp.json())
        else:
            st.error(f"Error: {resp.text}")
    if st.button("Receive and Validate Transaction as Bank 2"):
        resp = requests.get(f"{API_URL}/bank2/verify")
        if resp.status_code == 200:
            result = resp.json()
            st.success(f"Bank 2 received and verified transaction #{result['sample_id']}.")
            st.write(f"Prediction: {'FRAUD' if result['prediction'] else 'NOT FRAUD'}")
            st.write(f"ZKP Verified: {result['verified']}")
            if result.get('true_label') is not None:
                st.write(f"True Label: {result['true_label']}" )
        else:
            st.error(f"Error: {resp.text}")

st.header("Master Aggregator")
if st.button("Request Master Aggregator (Aggregate & ZKP)"):
    resp = requests.post(f"{API_URL}/federated/aggregate")
    if resp.status_code == 200:
        st.success("Master model aggregated and ZKP model saved.")
    else:
        st.error(f"Error: {resp.text}")
