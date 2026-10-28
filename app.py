import streamlit as st, pandas as pd, numpy as np, json
from xgboost import XGBClassifier

st.set_page_config(page_title="ML Network Intrusion Detection", layout="wide")
st.title("ML-based Network Intrusion Detection System")
st.caption("XGBoost classifier trained on CICIDS2017 (Mon-Wed). Upload network flow features to detect BENIGN, DoS and BruteForce traffic.")

@st.cache_resource
def load():
    m = XGBClassifier()
    m.load_model("xgb_model.json")
    return m, json.load(open("features.json")), json.load(open("classes.json"))

model, feats, classes = load()

use_sample = st.checkbox("Use the built-in sample file (300 flows)")
up = st.file_uploader("Or upload a CSV of flow features", type="csv")

df = None
if use_sample:
    df = pd.read_csv("sample_flows.csv")
elif up is not None:
    df = pd.read_csv(up, encoding="latin1", low_memory=False)

if df is not None:
    df.columns = df.columns.str.strip()
    missing = [c for c in feats if c not in df.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing[:5]) + " ...")
        st.stop()

    X = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    proba = model.predict_proba(X)
    out = df.copy()
    out["Prediction"] = np.array(classes)[proba.argmax(1)]
    out["Confidence %"] = (proba.max(1) * 100).round(2)

    alerts = out[out["Prediction"] != "BENIGN"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Flows analysed", len(out))
    c2.metric("Alerts", len(alerts))
    c3.metric("Benign", len(out) - len(alerts))

    st.subheader("Traffic breakdown")
    st.bar_chart(out["Prediction"].value_counts())

    st.subheader("Alerts")
    cols = ["Prediction", "Confidence %"] + feats[:6]
    st.dataframe(alerts[cols], use_container_width=True)

    st.download_button("Download full results", out.to_csv(index=False), "results.csv")
