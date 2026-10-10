import datetime as dt

import streamlit as st
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).parent / "cluster_classifier.pkl"

SPENDING_COLS = ["MntWines", "MntFruits", "MntMeatProducts",
                 "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
PURCHASE_COLS = ["NumDealsPurchases", "NumWebPurchases",
                 "NumCatalogPurchases", "NumStorePurchases"]
PROFILE_COLS = ["Income", "Age", "Total_Spending", "Total_Purchases",
                "Recency", "NumWebVisitsMonth"]
RAW_REQUIRED = (["Year_Birth", "Dt_Customer", "Income", "Kidhome", "Teenhome",
                 "Recency", "NumWebVisitsMonth", "Education", "Marital_Status"]
                + SPENDING_COLS + PURCHASE_COLS)


def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["meta"]


def engineer_features(raw, snapshot_date):
    """Same feature engineering as the notebook, applied to raw customer rows."""
    df = raw.copy()
    dates = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y", errors="coerce")
    df["Age"] = pd.Timestamp(snapshot_date).year - df["Year_Birth"]
    df["Tenure_Days"] = (pd.Timestamp(snapshot_date) - dates).dt.days
    df["Total_Spending"] = df[SPENDING_COLS].sum(axis=1)
    df["Total_Purchases"] = df[PURCHASE_COLS].sum(axis=1)
    return df


def segment_info(profile):
    """Name each cluster by ranking clusters on average spending (IDs are arbitrary)."""
    names_by_k = {
        2: ["Budget shoppers", "Premium spenders"],
        3: ["Budget shoppers", "Mainstream buyers", "Premium spenders"],
        4: ["Budget shoppers", "Occasional buyers", "Loyal regulars", "Premium spenders"],
    }
    order = profile["Total_Spending"].sort_values().index.tolist()
    names = names_by_k.get(len(order), [f"Spending tier {i + 1}" for i in range(len(order))])
    icons = ["🌱", "🛒", "⭐", "🔥", "🚀", "🏆", "👑", "💎"]
    info = {}
    for rank, cid in enumerate(order):
        row = profile.loc[cid]
        info[int(cid)] = {
            "name": names[rank],
            "icon": "💎" if rank == len(order) - 1 else icons[rank % len(icons)],
            "summary": (f"Avg income {row['Income']:,.0f} · avg spending "
                        f"{row['Total_Spending']:,.0f} · {row['Total_Purchases']:.0f} purchases"),
        }
    return info


def predict_one(model, row, num_cols, cat_cols):
    X = pd.DataFrame([row])[num_cols + cat_cols]
    probs = model.predict_proba(X)[0]
    classes = model.classes_
    return int(classes[int(np.argmax(probs))]), dict(zip(classes.astype(int), probs))


# =====================  STREAMLIT UI  =====================

st.set_page_config(page_title="Customer Segment Predictor", page_icon="🛍️", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
.hero {background: linear-gradient(135deg,#4f46e5 0%,#9333ea 100%);
       padding: 1.6rem 2rem; border-radius: 16px; color: white; margin-bottom: 1.2rem;}
.hero h1 {margin: 0; font-size: 2rem; color: white;}
.hero p {margin: .3rem 0 0 0; opacity: .9;}
.result {border-radius: 16px; padding: 1.4rem 1.6rem; border: 2px solid #9333ea;
         background: rgba(147,51,234,.08); margin: .5rem 0 1rem 0;}
.result h2 {margin: 0; font-size: 1.8rem;}
.result p {margin: .3rem 0 0 0; opacity: .8;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🛍️ Customer Segment Predictor</h1>
  <p>Enter a customer's details and find out which marketing segment they belong to.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def get_artifacts():
    return load_artifacts()


try:
    model, meta = get_artifacts()
except FileNotFoundError:
    st.error("`cluster_classifier.pkl` not found. Run the save cell in your notebook "
             "and put the file next to `app.py`.")
    st.stop()

num_cols, cat_cols = meta["num_cols"], meta["cat_cols"]
snapshot = meta["snapshot_date"]
profile = meta["cluster_profile"]
segments = segment_info(profile)
rng = meta["input_ranges"]
HIDDEN_MARITAL = {"Absurd", "YOLO"}  # junk categories in the raw data
marital_options = [m for m in meta["marital_options"] if m not in HIDDEN_MARITAL]

# ---------- Sidebar ----------
with st.sidebar:
    st.header("ℹ️ About this model")
    st.write(f"**Segments:** {len(segments)}")
    st.write(f"**Reference date:** {snapshot:%d %b %Y}")
    st.caption("A Random Forest trained to reproduce KMeans customer segments. "
               "Segments describe *behaviour*, not who will respond to a campaign.")
    st.divider()
    for cid, s in segments.items():
        st.markdown(f"**{s['icon']} {s['name']}**  \n{s['summary']}")

tab_single, tab_profiles, tab_batch = st.tabs(
    ["🔮 Predict a customer", "📊 Segment profiles", "📁 Batch prediction"])

# ---------- Tab 1: single customer ----------
with tab_single:
    with st.form("customer_form"):
        st.subheader("Customer profile")
        c1, c2, c3 = st.columns(3)
        income = c1.number_input("Annual income", 0.0, float(rng["Income"][1]),
                                 float(rng["Income"][2]), step=1000.0)
        age = c2.slider("Age", rng["Age"][0], rng["Age"][1], rng["Age"][2])
        joined = c3.date_input("Customer since",
                               value=(snapshot - dt.timedelta(days=365)).date(),
                               min_value=dt.date(2010, 1, 1), max_value=snapshot.date())

        c4, c5, c6 = st.columns(3)
        education = c4.selectbox("Education", meta["education_options"],
                                 index=meta["education_options"].index("Graduation")
                                 if "Graduation" in meta["education_options"] else 0)
        marital = c5.selectbox("Marital status", marital_options)
        recency = c6.slider("Days since last purchase", 0, rng["Recency"][1], rng["Recency"][2])

        c7, c8, c9 = st.columns(3)
        kids = c7.number_input("Kids at home", 0, 5, 0)
        teens = c8.number_input("Teenagers at home", 0, 5, 0)
        web_visits = c9.slider("Website visits / month", 0, rng["NumWebVisitsMonth"][1],
                               rng["NumWebVisitsMonth"][2])

        st.subheader("Spending (last 2 years)")
        s = st.columns(6)
        wines = s[0].number_input("Wine", 0, 5000, 170)
        fruits = s[1].number_input("Fruits", 0, 5000, 8)
        meat = s[2].number_input("Meat", 0, 5000, 65)
        fish = s[3].number_input("Fish", 0, 5000, 12)
        sweets = s[4].number_input("Sweets", 0, 5000, 8)
        gold = s[5].number_input("Gold products", 0, 5000, 24)

        st.subheader("Purchases by channel")
        p = st.columns(4)
        deals = p[0].number_input("Deals", 0, 50, 2)
        web = p[1].number_input("Website", 0, 50, 4)
        catalog = p[2].number_input("Catalog", 0, 50, 2)
        store = p[3].number_input("Store", 0, 50, 5)

        submitted = st.form_submit_button("Predict segment", type="primary",
                                          use_container_width=True)

    if submitted:
        row = {
            "Income": income, "Age": age,
            "Tenure_Days": (snapshot.date() - joined).days,
            "Kidhome": kids, "Teenhome": teens, "Recency": recency,
            "NumWebVisitsMonth": web_visits,
            "Total_Spending": wines + fruits + meat + fish + sweets + gold,
            "Total_Purchases": deals + web + catalog + store,
            "Education": education, "Marital_Status": marital,
        }
        best, probs = predict_one(model, row, num_cols, cat_cols)
        seg = segments[best]

        st.markdown(f"""
        <div class="result">
          <h2>{seg['icon']} {seg['name']}</h2>
          <p>Segment {best} · {seg['summary']}</p>
        </div>
        """, unsafe_allow_html=True)

        left, right = st.columns([1, 1.4])
        with left:
            st.metric("Model confidence", f"{probs[best]:.0%}")
            st.caption("Probability the model assigns to this segment.")
            chart = pd.DataFrame({"Probability": {segments[c]["name"]: v for c, v in probs.items()}})
            st.bar_chart(chart)
        with right:
            st.markdown("**This customer vs. the segment average**")
            avg = profile.loc[best]
            mcols = st.columns(3)
            for i, col in enumerate(PROFILE_COLS):
                value = row[col]
                mcols[i % 3].metric(col.replace("_", " "), f"{value:,.0f}",
                                    f"{value - avg[col]:+,.0f} vs avg")

# ---------- Tab 2: segment profiles ----------
with tab_profiles:
    st.subheader("What each segment looks like")
    cols = st.columns(len(segments))
    for col, (cid, s) in zip(cols, segments.items()):
        with col:
            st.markdown(f"### {s['icon']} {s['name']}")
            st.caption(f"Segment {cid} · {int(profile.loc[cid, 'Customers']):,} customers")
            st.write(s["summary"])
    st.markdown("**Average values per segment**")
    shown = profile.rename(index={cid: f"{s['icon']} {s['name']}" for cid, s in segments.items()})
    st.dataframe(shown, use_container_width=True)
    st.markdown("**Compare a metric across segments**")
    metric = st.selectbox("Metric", PROFILE_COLS, format_func=lambda c: c.replace("_", " "))
    st.bar_chart(shown[metric])

# ---------- Tab 3: batch ----------
with tab_batch:
    st.subheader("Score a whole customer file")
    st.write("Upload a CSV in the **same raw format as the original dataset** "
             "(tab- or comma-separated). Age, tenure and totals are calculated for you.")
    up = st.file_uploader("Customer CSV", type=["csv", "tsv", "txt"])
    if up is not None:
        raw = pd.read_csv(up, sep=None, engine="python")
        missing = [c for c in RAW_REQUIRED if c not in raw.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            feats = engineer_features(raw, snapshot)
            preds = model.predict(feats[num_cols + cat_cols])
            out = raw.copy()
            out["Segment"] = [segments[int(p)]["name"] for p in preds]
            bad_dates = feats["Tenure_Days"].isna().sum()
            if bad_dates:
                st.warning(f"{bad_dates} row(s) had an unreadable date (expected DD-MM-YYYY); "
                           "the model filled the gap with a typical value.")
            m1, m2 = st.columns([1, 2])
            m1.metric("Customers scored", f"{len(out):,}")
            m2.bar_chart(out["Segment"].value_counts())
            st.dataframe(out.head(100), use_container_width=True)
            st.download_button("⬇️ Download results (CSV)",
                               out.to_csv(index=False).encode("utf-8"),
                               "segmented_customers.csv", "text/csv")


