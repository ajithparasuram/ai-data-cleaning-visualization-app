# app.py
import warnings
warnings.filterwarnings("ignore", message="Please replace `use_container_width`")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# optional sklearn features
try:
    from sklearn.impute import KNNImputer
    from sklearn.ensemble import IsolationForest
    SKLEARN = True
except Exception:
    SKLEARN = False

# ---------- Page config ----------
st.set_page_config(page_title="AI Data Cleaner & Visualizer", layout="wide")
st.title("🧠 AI Data Cleaning & Visualization App")

# ---------- Helper functions ----------
def dataset_summary(df):
    """Return a small summary dict and a dataframe of column stats."""
    total_rows = df.shape[0]
    col_stats = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing = int(df[col].isna().sum())
        missing_pct = round(100 * missing / total_rows, 2) if total_rows > 0 else 0
        unique = int(df[col].nunique(dropna=True))
        top = df[col].mode().iloc[0] if unique > 0 else ""
        col_stats.append({
            "column": col,
            "dtype": dtype,
            "unique": unique,
            "missing": missing,
            "missing_pct": missing_pct,
            "top_value": top
        })
    stats_df = pd.DataFrame(col_stats)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = None
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
    return {"rows": total_rows, "cols": df.shape[1], "numeric_count": len(numeric_cols)}, stats_df, corr

def clean_dataframe(df, impute_method="Simple", neighbors=5,
                    remove_outliers=False, outlier_frac=0.01):
    report = {}

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = before - len(df)

    # Count missing before
    report["missing_before"] = int(df.isna().sum().sum())

    # Imputation
    if impute_method == "KNN" and SKLEARN:
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

    # Outlier removal (IsolationForest)
    report["outliers_removed"] = 0
    if remove_outliers and SKLEARN:
        numeric = df.select_dtypes(include=[np.number])
        if not numeric.empty:
            iso = IsolationForest(contamination=outlier_frac, random_state=42)
            preds = iso.fit_predict(numeric)
            mask = preds != -1
            report["outliers_removed"] = int((~mask).sum())
            df = df.loc[mask].reset_index(drop=True)

    return df, report

# ---------- File upload ----------
uploaded = st.file_uploader("Upload a CSV file", type=["csv", "txt"])
if uploaded is None:
    st.info("👆 Upload a CSV to get started. Example: cleaned_data.csv")
    st.stop()

# read CSV safely (support for different encodings)
try:
    df = pd.read_csv(uploaded)
except Exception:
    df = pd.read_csv(uploaded, encoding="latin1")

st.success("File loaded successfully!")
st.write("### Raw data preview", df.head(6))

# ---------- AI-style data summary ----------
def generate_text_summary(df):
    rows, cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    missing_total = df.isna().sum().sum()
    missing_cols = df.columns[df.isna().any()].tolist()

    summary = f"Your dataset has **{rows:,} rows** and **{cols} columns**. "
    summary += f"It includes **{len(numeric_cols)} numeric** and **{len(cat_cols)} categorical** columns. "

    if missing_total > 0:
        summary += f"There are **{missing_total:,} missing values** across **{len(missing_cols)} columns**. "
        worst = df.isna().sum().sort_values(ascending=False).head(1)
        col = worst.index[0]
        pct = (worst.iloc[0] / rows) * 100
        summary += f"The column with most missing data is **{col} ({pct:.1f}% missing)**. "
    else:
        summary += "There are **no missing values** in this dataset. "

    # Find top categorical column
    if len(cat_cols) > 0:
        top_col = cat_cols[0]
        top_val = df[top_col].mode().iloc[0] if df[top_col].notna().any() else "N/A"
        summary += f"The most frequent category in **{top_col}** is **{top_val}**. "

    # Range of numeric values
    if len(numeric_cols) > 0:
        mean_vals = df[numeric_cols].mean().sort_values(ascending=False)
        top_num = mean_vals.index[0]
        summary += f"Among numeric columns, **{top_num}** has the highest average value ({mean_vals.iloc[0]:.2f})."

    return summary

st.write("### 🧠 Data Summary")
st.markdown(generate_text_summary(df))

# ---------- Dataset summary (Tableau-like quick profile) ----------
summary_info, stats_df, corr = dataset_summary(df)
col1, col2, col3 = st.columns(3)
col1.metric("Rows", summary_info["rows"])
col2.metric("Columns", summary_info["cols"])
col3.metric("Numeric cols", summary_info["numeric_count"])

with st.expander("Show column-level summary"):
    st.dataframe(stats_df, use_container_width=True)

if corr is not None:
    with st.expander("Numeric correlation (heatmap)"):
        fig_corr = px.imshow(corr, text_auto=True)
        st.plotly_chart(fig_corr, width='stretch')

# ---------- Sidebar cleaning options ----------
st.sidebar.header("🧹 Auto-Clean Options")
impute_choice = st.sidebar.selectbox("Imputation method",
                                     ["Simple (median/mode)", "KNN (ML-based)"])
if "KNN" in impute_choice and not SKLEARN:
    st.sidebar.warning("KNN imputer requires scikit-learn. Install scikit-learn to enable.")
neighbors = st.sidebar.slider("KNN neighbors", 1, 15, 5)
remove_outliers = st.sidebar.checkbox("Remove numeric outliers (IsolationForest)", value=False)
outlier_frac = st.sidebar.slider("Outlier fraction", 0.001, 0.2, 0.01, 0.001)
remove_duplicates = st.sidebar.checkbox("Remove duplicates (already applied in clean)", value=True)

# ---------- Run cleaning ----------
if st.sidebar.button("Run Auto-Clean"):
    with st.spinner("Cleaning data..."):
        df_clean, report = clean_dataframe(
            df,
            impute_method="KNN" if "KNN" in impute_choice else "Simple",
            neighbors=neighbors,
            remove_outliers=remove_outliers,
            outlier_frac=outlier_frac,
        )
    st.success("✅ Cleaning complete!")
    st.write("### Cleaning summary")
    st.json(report)
    st.write("### Cleaned data sample", df_clean.head())
    csv = df_clean.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download cleaned CSV", csv, "cleaned_data.csv", "text/csv")
    df_to_plot = df_clean
else:
    df_to_plot = df.copy()

# ---------- Quick filters (slice data before plotting) ----------
st.sidebar.header("🔎 Quick Filters (slice data)")
filter_cols = df_to_plot.columns.tolist()
f1 = st.sidebar.selectbox("Filter column 1 (optional)", [""] + filter_cols, index=0)
f2 = st.sidebar.selectbox("Filter column 2 (optional)", [""] + filter_cols, index=0)

filtered = df_to_plot
if f1:
    if pd.api.types.is_numeric_dtype(filtered[f1]):
        minv, maxv = float(filtered[f1].min()), float(filtered[f1].max())
        sel = st.sidebar.slider(f"{f1} range", minv, maxv, (minv, maxv))
        filtered = filtered[filtered[f1].between(sel[0], sel[1])]
    else:
        vals = filtered[f1].unique().tolist()
        sel = st.sidebar.multiselect(f"Select {f1}", vals, default=vals[:3])
        if sel:
            filtered = filtered[filtered[f1].isin(sel)]
if f2:
    if pd.api.types.is_numeric_dtype(filtered[f2]):
        minv, maxv = float(filtered[f2].min()), float(filtered[f2].max())
        sel = st.sidebar.slider(f"{f2} range", minv, maxv, (minv, maxv))
        filtered = filtered[filtered[f2].between(sel[0], sel[1])]
    else:
        vals = filtered[f2].unique().tolist()
        sel = st.sidebar.multiselect(f"Select {f2}", vals, default=vals[:3])
        if sel:
            filtered = filtered[filtered[f2].isin(sel)]

# ---------- Visualization ----------
st.write("---")
st.header("📊 Visualizations")

if filtered.empty:
    st.warning("No data to visualize (filters removed all rows).")
    st.stop()

cols = filtered.columns.tolist()
xcol = st.selectbox("Select column to visualize (X)", cols)

plot_type = st.radio("Choose chart type",
                     ["Histogram", "Box", "Bar (counts)", "Scatter"],
                     horizontal=True)

if plot_type == "Histogram":
    fig = px.histogram(filtered, x=xcol)
    st.plotly_chart(fig, width='stretch')

elif plot_type == "Box":
    fig = px.box(filtered, y=xcol)
    st.plotly_chart(fig, width='stretch')

elif plot_type == "Bar (counts)":
    vc = filtered[xcol].value_counts().reset_index()
    new_x = f"{xcol}_value"
    new_y = f"{xcol}_frequency"
    vc.columns = [new_x, new_y]
    fig = px.bar(vc, x=new_x, y=new_y, title=f"Value counts of {xcol}")
    st.plotly_chart(fig, width='stretch')

elif plot_type == "Scatter":
    num_cols = filtered.select_dtypes(include=[np.number]).columns.tolist()
    if xcol not in num_cols:
        st.info("Select a numeric column for scatter.")
    else:
        ycol = st.selectbox("Y-axis column (numeric)", [c for c in num_cols if c != xcol])
        fig = px.scatter(filtered, x=xcol, y=ycol)
        st.plotly_chart(fig, width='stretch')

st.write("---")
st.caption("AI Data Cleaner © 2025 — local processing only, no data uploaded.")