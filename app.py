# app.py
# Polished AI Data Cleaner & Visualizer — Ajith Parasuram

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
import io, base64

# Page config
st.set_page_config(
    page_title="AI Data Cleaner & Visualizer — Ajith",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS + helpers
st.markdown(
    """
    <style>
      .big-title {font-size:30px; font-weight:700; color:#0f172a;}
      .muted {color:#6b7280; font-size:13px;}
      .card {background: #ffffff; border-radius:10px; padding:12px; box-shadow: 0 2px 6px rgba(15,23,42,0.05);}
      .footer {color: #94a3b8; font-size:12px; margin-top:20px;}
      .small {font-size:12px; color:#475569;}
      a {color: #0f172a;}
    </style>
    """,
    unsafe_allow_html=True,
)

def footer():
    st.markdown("---")
    st.markdown(
        f"""<div class="footer">Made by **Ajith Parasuram** — <a href="https://github.com/ajithparasuram" target="_blank">GitHub</a> • <a href="mailto:ajithparasuram111@gmail.com">Email</a></div>""",
        unsafe_allow_html=True,
    )

def make_download_button(df, filename="cleaned_data.csv", label="⬇️ Download cleaned CSV"):
    """Return a Streamlit download_button for a DataFrame"""
    csv = df.to_csv(index=False).encode("utf-8")
    return st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

def sample_dataset_bytes():
    df = pd.DataFrame({
        "PassengerId":[1,2,3,4,5],
        "Survived":[0,1,1,1,0],
        "Pclass":[3,1,3,1,3],
        "Name":["A","B","C","D","E"],
        "Age":[22,38,26,35,35],
        "Fare":[7.25,71.2833,7.925,53.1,8.05]
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf

# Title area
col1, col2 = st.columns([0.12, 0.88])
with col1:
    st.image("https://static.streamlit.io/examples/book-cover.png", width=64)
with col2:
    st.markdown('<div class="big-title">AI Data Cleaner & Visualizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Upload CSV → Auto-clean (KNN or median/mode) → Outlier removal → Interactive charts → Download cleaned CSV</div>', unsafe_allow_html=True)
st.write("")

# ---------------- Sidebar controls ----------------
st.sidebar.header("⚙️ Controls")
st.sidebar.markdown("**Upload** data and configure cleaning options here.")
uploaded = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"], help="Upload a CSV or Excel file (first row = headers)")

st.sidebar.markdown("### Auto-clean options")
impute_choice = st.sidebar.selectbox("Imputation method",
                                     ["Simple (median/mode)", "KNN (ML-based)"])
neighbors = st.sidebar.slider("KNN neighbors", 1, 15, 5)
remove_outliers = st.sidebar.checkbox("Remove numeric outliers (IsolationForest)", value=False)
outlier_frac = st.sidebar.slider("Outlier fraction", 0.001, 0.2, 0.01, 0.001)

st.sidebar.markdown("---")
st.sidebar.markdown("**Quick actions**")
if st.sidebar.button("Use sample dataset"):
    uploaded = sample_dataset_bytes()
if st.sidebar.button("Run Auto-Clean"):
    st.session_state.run_clean = True
else:
    if "run_clean" not in st.session_state:
        st.session_state.run_clean = False

st.sidebar.markdown("---")
st.sidebar.markdown("**Help**")
st.sidebar.info("Tip: Use the sample dataset to try the app quickly. Do not upload sensitive or private data on public demos.")

# ---------------- Cleaning functions ----------------
def clean_dataframe(df, impute_method="Simple", neighbors=5,
                    remove_outliers=False, outlier_frac=0.01):
    report = {}
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = before - len(df)
    report["missing_before"] = int(df.isna().sum().sum())

    # Imputation
    if impute_method == "KNN":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            imputer = KNNImputer(n_neighbors=neighbors)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    else:
        for col in df.columns:
            if df[col].dtype.kind in "biufc":
                df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode()
                if len(mode) > 0:
                    df[col] = df[col].fillna(mode[0])
                else:
                    df[col] = df[col].fillna("")

    report["missing_after"] = int(df.isna().sum().sum())
    report["outliers_removed"] = 0
    if remove_outliers:
        numeric = df.select_dtypes(include=[np.number])
        if not numeric.empty:
            iso = IsolationForest(contamination=outlier_frac, random_state=42)
            preds = iso.fit_predict(numeric)
            mask = preds != -1
            report["outliers_removed"] = int((~mask).sum())
            df = df.loc[mask].reset_index(drop=True)
    return df, report

# ---------------- Load file & run cleaning ----------------
if uploaded is None:
    st.info("👆 Upload a CSV to get started or use the Sample dataset.")
    st.stop()

# read file
if hasattr(uploaded, "read"):
    try:
        if str(uploaded).lower().endswith(".xlsx"):
            df = pd.read_excel(uploaded)
        else:
            # try normal csv
            df = pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        df = pd.read_csv(uploaded, encoding="latin1")
else:
    # sample dataset bytesIO case
    df = pd.read_csv(uploaded)

st.success("File loaded successfully!")
st.write("### Data preview")
st.dataframe(df.head())

# Run cleaning if requested
if st.session_state.run_clean:
    with st.spinner("Cleaning data..."):
        df_clean, report = clean_dataframe(
            df,
            impute_method="KNN" if "KNN" in impute_choice else "Simple",
            neighbors=neighbors,
            remove_outliers=remove_outliers,
            outlier_frac=outlier_frac,
        )
    st.success("✅ Cleaning complete")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df_clean.shape[0]:,}")
    c2.metric("Columns", f"{df_clean.shape[1]}")
    c3.metric("Missing after", f"{int(df_clean.isna().sum().sum())}")

    st.markdown("#### Cleaning summary")
    st.json(report)

    st.markdown("#### Cleaned data sample")
    st.dataframe(df_clean.head())

    # download button
    make_download_button(df_clean, filename="cleaned_data.csv", label="⬇️ Download cleaned CSV")

    df_to_plot = df_clean
else:
    df_to_plot = df

# ---------------- Visualizations ----------------
st.write("---")
st.header("📊 Visualizations")

if df_to_plot is None or df_to_plot.empty:
    st.warning("No data to visualize. Run cleaning or upload a file.")
    st.stop()

cols = df_to_plot.columns.tolist()
xcol = st.selectbox("Select column to visualize", cols, index=0)
plot_type = st.radio("Choose chart type", ["Histogram", "Box", "Bar (counts)", "Scatter"], horizontal=True)

with st.container():
    if plot_type == "Histogram":
        fig = px.histogram(df_to_plot, x=xcol, nbins=30, title=f"Histogram — {xcol}")
        st.plotly_chart(fig, use_container_width=True)
    elif plot_type == "Box":
        fig = px.box(df_to_plot, y=xcol, title=f"Box plot — {xcol}")
        st.plotly_chart(fig, use_container_width=True)
    elif plot_type == "Bar (counts)":
        vc = df_to_plot[xcol].value_counts().reset_index()
        vc.columns = [xcol, "count"]
        fig = px.bar(vc, x=xcol, y="count", title=f"Value counts — {xcol}")
        st.plotly_chart(fig, use_container_width=True)
    elif plot_type == "Scatter":
        numeric_cols = df_to_plot.select_dtypes(include=[np.number]).columns.tolist()
        if xcol not in numeric_cols:
            st.info("Select a numeric column for scatter.")
        else:
            ycol = st.selectbox("Y-axis column (numeric)", [c for c in numeric_cols if c != xcol])
            if ycol:
                fig = px.scatter(df_to_plot, x=xcol, y=ycol, title=f"Scatter — {xcol} vs {ycol}")
                st.plotly_chart(fig, use_container_width=True)

# ---------------- About & Footer ----------------
st.write("---")
with st.expander("About this app / How to use"):
    st.markdown("""
    **AI Data Cleaner & Visualizer**  
    *Author:* Ajith Parasuram  
    *Stack:* Python, Streamlit, Pandas, scikit-learn, Plotly  
    **How to use:**  
    1. Upload a CSV (first row should be headers).  
    2. Choose imputation method (Simple / KNN), set K for KNN.  
    3. (Optional) Toggle outlier removal.  
    4. Click **Run Auto-Clean** in the sidebar.  
    5. View cleaning summary, download cleaned CSV, and explore visualizations.  
    **Note:** This demo processes your data locally; do not upload sensitive production data to public demos.  
    """)

footer()