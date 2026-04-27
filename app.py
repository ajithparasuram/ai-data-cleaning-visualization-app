import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
import io

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Data Cleaner", layout="wide")

# ---------------- STATE ----------------
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None

# ---------------- HEADER ----------------
st.title("🧠 AI Data Cleaner & Visualizer")
st.caption("Upload → Clean → Profile → Visualize → Download")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")

uploaded = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

# Sample dataset
if st.sidebar.button("Use sample dataset"):
    df_sample = pd.DataFrame({
        "A":[1,2,3,np.nan,5],
        "B":[10,20,np.nan,40,50],
        "C":["x","y","x","z",None]
    })
    buf = io.BytesIO()
    df_sample.to_csv(buf, index=False)
    buf.seek(0)
    uploaded = buf

# ---------------- LOAD FUNCTION ----------------
def load_data(file):
    try:
        if file is None:
            return None
        if hasattr(file, "name") and file.name.endswith(".xlsx"):
            return pd.read_excel(file)
        return pd.read_csv(file)
    except:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding="latin1")
        except:
            return None

df = load_data(uploaded)

# ---------------- FALLBACK ----------------
if df is None:
    st.warning("⚠️ No file uploaded → using sample dataset")
    df = pd.DataFrame({
        "A":[1,2,3,np.nan,5],
        "B":[10,20,np.nan,40,50],
        "C":["x","y","x","z",None]
    })

# ---------------- DISPLAY ----------------
st.success("✅ Data ready")
st.subheader("📄 Data Preview")
st.dataframe(df.head())

# ---------------- OPTIONS ----------------
st.sidebar.markdown("### Cleaning Options")

impute_choice = st.sidebar.selectbox("Imputation", ["Simple", "KNN"])
neighbors = st.sidebar.slider("KNN neighbors", 1, 15, 5)

remove_outliers = st.sidebar.checkbox("Remove outliers")
outlier_frac = st.sidebar.slider("Outlier fraction", 0.001, 0.2, 0.01)

run_clean = st.sidebar.button("Run Auto-Clean")

# ---------------- CLEAN FUNCTION ----------------
def clean_dataframe(df):
    df = df.copy()

    # Fill missing values
    if impute_choice == "KNN":
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            imputer = KNNImputer(n_neighbors=neighbors)
            df[num_cols] = imputer.fit_transform(df[num_cols])
    else:
        for col in df.columns:
            if df[col].dtype.kind in "biufc":
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "")

    # Remove outliers
    if remove_outliers:
        numeric = df.select_dtypes(include=[np.number])
        if not numeric.empty:
            iso = IsolationForest(contamination=outlier_frac)
            preds = iso.fit_predict(numeric)
            df = df[preds != -1]

    return df

# ---------------- CLEAN ----------------
if run_clean:
    st.session_state.df_clean = clean_dataframe(df)

# Use cleaned or raw
df_plot = st.session_state.df_clean if st.session_state.df_clean is not None else df

# ---------------- DATA PROFILING ----------------
st.markdown("---")
st.header("📊 Data Profiling")

st.subheader("Summary Statistics")
st.dataframe(df_plot.describe())

st.subheader("Missing Values")
missing = df_plot.isna().sum()
st.dataframe(missing[missing > 0])

# ---------------- VISUALIZATION ----------------
st.markdown("---")
st.header("📊 Visualization")

cols = df_plot.columns.tolist()
xcol = st.selectbox("Select column", cols)

chart = st.radio(
    "Chart Type",
    ["Histogram", "Box", "Bar", "Scatter"],
    horizontal=True
)

if chart == "Histogram":
    fig = px.histogram(df_plot, x=xcol)
    st.plotly_chart(fig, width="stretch")

elif chart == "Box":
    fig = px.box(df_plot, y=xcol)
    st.plotly_chart(fig, width="stretch")

elif chart == "Bar":
    vc = df_plot[xcol].value_counts().reset_index()
    vc.columns = [xcol, "count"]
    fig = px.bar(vc, x=xcol, y="count")
    st.plotly_chart(fig, width="stretch")

elif chart == "Scatter":
    num_cols = df_plot.select_dtypes(include=['number']).columns
    if len(num_cols) > 1:
        ycol = st.selectbox("Select Y-axis", num_cols)
        fig = px.scatter(df_plot, x=xcol, y=ycol)
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("Need at least 2 numeric columns for scatter plot")

# ---------------- DOWNLOAD ----------------
if st.session_state.df_clean is not None:
    csv = df_plot.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Cleaned Data", csv, "cleaned.csv")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built by Ajith 🚀")
