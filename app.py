# Streamlit — MetaboAnalyst Post‑Plots (PCA, PLS‑DA, VIP)
# -------------------------------------------------------
# Minimal, robust scheme to load MetaboAnalyst score/VIP CSVs
# and generate publication‑ready figures with confidence ellipses.
#
# IMPORTANT: This version **does NOT require a separate metadata.csv**.
# You can:
#   • use a grouping column that already exists in the score CSVs (e.g., Group/Class), or
#   • derive groups from Sample names via a regex rule.
#
# How to run locally:
#   1) pip install streamlit plotly pandas numpy scipy matplotlib
#   2) streamlit run metaboanalyst_postplots_streamlit_app.py

from __future__ import annotations
import os
import io
import re
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import streamlit as st
st.set_page_config(page_title="MetaboAnalyst Post-Plots", layout="wide")

import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import chi2
from pathlib import Path
from PIL import Image, UnidentifiedImageError

# ------------------ LOGO ------------------

STATIC_DIR = Path(__file__).parent / "static"
LOGO_PATH = STATIC_DIR / "LAABio.png"

try:
    logo = Image.open(LOGO_PATH)  # raises if missing
    st.sidebar.image(logo, use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning("Logo not found at static/LAABio.png")

st.markdown(
    """
    
    Developed by **Ricardo M Borges** and **LAABio-IPPN-UFRJ**  
    contact: ricardo_mborges@yahoo.com.br  

    🔗 Details: [GitHub repository](https://github.com/RicardoMBorges/new_figure_metaboAnalyst)

    Check also: [DAFdiscovery](https://dafdiscovery.streamlit.app/)
    
    Check also: [TLC2Chrom](https://tlc2chrom.streamlit.app/)
    """
)

# PayPal donate button
st.sidebar.markdown("""
<hr>
<center>
<p>To support the app development:</p>
<a href="https://www.paypal.com/donate/?business=2FYTFNDV4F2D4&no_recurring=0&item_name=Support+with+%245+→+Send+receipt+to+tlc2chrom.app@gmail.com+with+your+login+email+→+Access+within+24h!&currency_code=USD" target="_blank">
    <img src="https://www.paypalobjects.com/en_US/i/btn/btn_donate_SM.gif" alt="Donate with PayPal button" border="0">
</a>
</center>
""", unsafe_allow_html=True)

st.sidebar.markdown("""---""")

TUTORIAL_URL = "https://github.com/RicardoMBorges/new_figure_metaboAnalyst/blob/main/README.md"
try:
    st.sidebar.link_button("📘 Tutorial", TUTORIAL_URL)
except Exception:
    st.sidebar.markdown(
        f'<a href="{TUTORIAL_URL}" target="_blank">'
        '<button style="padding:0.6rem 1rem; border-radius:8px; border:1px solid #ddd; cursor:pointer;">📘 Tutorial</button>'
        '</a>',
        unsafe_allow_html=True,
    )


MockData_URL = "https://github.com/RicardoMBorges/new_figure_metaboAnalyst/tree/main/Tutorial_Data"
try:
    st.sidebar.link_button("Tutorial Data", MockData_URL)
except Exception:
    st.sidebar.markdown(
        f'<a href="{MockData_URL}" target="_blank">'
        '<button style="padding:0.6rem 1rem; border-radius:8px; border:1px solid #ddd; cursor:pointer;">Mock Data</button>'
        '</a>',
        unsafe_allow_html=True,
    )
    
VIDEO_URL = "https://github.com/RicardoMBorges/new_figure_metaboAnalyst"
try:
    st.sidebar.link_button("Video", VIDEO_URL)
except Exception:
    st.sidebar.markdown(
        f'<a href="{VIDEO_URL}" target="_blank">'
        '<button style="padding:0.6rem 1rem; border-radius:8px; border:1px solid #ddd; cursor:pointer;">📘 Tutorial</button>'
        '</a>',
        unsafe_allow_html=True,
    )

st.sidebar.markdown("""---""")


# ------------------------------
# ---------- Utilities ---------
# ------------------------------

@st.cache_data
def read_csv_any(buf, **kw) -> pd.DataFrame:
    try:
        return pd.read_csv(buf, **kw)
    except Exception:
        # Fallback: try semicolon
        try:
            if hasattr(buf, "seek"): buf.seek(0)
            return pd.read_csv(buf, sep=';', engine='python', **{k:v for k,v in kw.items() if k != 'sep'})
        except Exception:
            if isinstance(buf, (str, os.PathLike)):
                return pd.read_csv(buf)
            raise


def try_guess_col(df: pd.DataFrame, prefixes: List[str], default: Optional[str]=None) -> Optional[str]:
    pref = tuple(p.lower() for p in prefixes)
    for c in df.columns:
        cl = str(c).lower()
        if cl.startswith(pref):
            return c
    return default


def ensure_sample_col(df: pd.DataFrame) -> pd.DataFrame:
    # MetaboAnalyst usually places sample IDs in the first column named 'Unnamed: 0'.
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "Sample"})
    else:
        first = df.columns[0]
        if str(first).lower() not in {"sample", "samples"}:
            df = df.rename(columns={first: "Sample"})
    return df


def build_group_from_regex(series: pd.Series, pattern: str, default_label: str="Group") -> pd.Series:
    """Extract group labels from sample names.
    pattern should contain **one** capturing group, e.g. r"^([^_]+)" or r"^(.*?)-rep".
    """
    rgx = re.compile(pattern)
    out = []
    for s in series.astype(str):
        m = rgx.search(s)
        out.append(m.group(1) if m else default_label)
    return pd.Series(out, index=series.index, name="Group")


def ellipse_path(x, y, conf: float=0.90, n: int=200) -> Optional[str]:
    x = np.asarray(x); y = np.asarray(y)
    if len(x) < 3:
        return None
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)):
        return None
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    k = np.sqrt(chi2.ppf(conf, 2))
    t = np.linspace(0, 2*np.pi, n)
    circ = np.stack((np.cos(t), np.sin(t)))
    ell = (vecs @ (np.sqrt(np.maximum(vals, 0))[:, None] * circ)) * k
    cx, cy = float(np.mean(x)), float(np.mean(y))
    xe, ye = ell[0, :] + cx, ell[1, :] + cy
    path = "M " + " L ".join(f"{xx},{yy}" for xx, yy in zip(xe, ye)) + " Z"
    return path

# Line-only ellipse (no fill) coordinates for Plotly scatter

def ellipse_xy(x, y, conf: float = 0.90, n: int = 200):
    x = np.asarray(x); y = np.asarray(y)
    if len(x) < 3:
        return None
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)):
        return None
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    k = np.sqrt(chi2.ppf(conf, 2))
    t = np.linspace(0, 2*np.pi, n)
    circ = np.stack((np.cos(t), np.sin(t)))
    ell = (vecs @ (np.sqrt(np.maximum(vals, 0))[:, None] * circ)) * k
    cx, cy = float(np.mean(x)), float(np.mean(y))
    xe, ye = ell[0, :] + cx, ell[1, :] + cy
    return xe, ye

# ---------------- R2/Q2 helpers ----------------

def _canon_id(s: str, strip_ext: bool = True) -> str:
    s = str(s).strip()
    if strip_ext:
        s = re.sub(r"\.[A-Za-z0-9]+$", "", s)  # drop .csv/.mzML/.txt etc.
    return s.lower()  # case-insensitive match

def prepare_X_from_normalized(df: pd.DataFrame, sample_names: List[str]) -> Optional[pd.DataFrame]:
    """Return X (rows=samples, cols=features) aligned to sample_names.
    Tries multiple common MetaboAnalyst orientations. Robust to case/whitespace/extension differences.
    """
    df = df.copy()
    # Canonicalize
    sample_names = [ _canon_id(s) for s in sample_names ]
    df.columns = [ str(c).strip() for c in df.columns ]
    if df.columns[0].startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: "ID"})

    def _align_by_columns(frame: pd.DataFrame) -> Optional[pd.DataFrame]:
        cols_map = { _canon_id(c): c for c in frame.columns }
        overlap_keys = [k for k in sample_names if k in cols_map]
        if len(overlap_keys) >= max(2, int(0.3 * len(sample_names))):
            ordered_cols = [ cols_map[k] for k in sample_names if k in cols_map ]
            X = frame[ordered_cols].T  # samples x features
            X.index = [ _canon_id(ix, False) for ix in X.index ]
            # Reindex to canonical sample_names order (already ordered)
            X = X.reindex([ _canon_id(s, False) for s in ordered_cols ])  # not strictly needed
            return X.select_dtypes(include=[np.number])
        return None

    def _align_by_index(frame: pd.DataFrame, index_col: str) -> Optional[pd.DataFrame]:
        f = frame.copy()
        f[index_col] = f[index_col].astype(str).str.strip()
        f["_canon"] = f[index_col].apply(_canon_id)
        f = f.set_index("_canon")
        inter = f.index.intersection(sample_names)
        if len(inter) >= max(2, int(0.3 * len(sample_names))):
            X = f.loc[sample_names].drop(columns=[c for c in [index_col] if c in f.columns], errors="ignore")
            return X.select_dtypes(include=[np.number])
        return None

    lower_cols = [c.lower() for c in df.columns]

    # Case A: samples are rows (a column named Sample/Samples)
    if "sample" in lower_cols or "samples" in lower_cols:
        sc = df.columns[lower_cols.index("sample")] if "sample" in lower_cols else df.columns[lower_cols.index("samples")]
        X = _align_by_index(df, sc)
        if X is not None and not X.empty:
            return X

    # Case B: samples are columns (features x samples)
    X = _align_by_columns(df)
    if X is not None and not X.empty:
        return X

    # Case C: first col is feature IDs; remaining cols might be samples
    X = _align_by_columns(df.iloc[:, 1:])
    if X is not None and not X.empty:
        return X

    # Case D: transpose as last resort
    dft = df.T
    dft["__rowid__"] = dft.index
    X = _align_by_index(dft.reset_index(drop=True), "__rowid__")
    if X is not None and not X.empty:
        return X

    return None


def compute_pls_r2q2(X: pd.DataFrame, y: pd.Series, max_comps: int = 5, cv_folds: int = 7):
    """Compute R2Y and Q2 across 1..max_comps using sklearn's PLSRegression and KFold CV.
    Returns a DataFrame with columns: ncomp, R2Y, Q2.
    """
    try:
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import KFold
        from sklearn.metrics import r2_score
    except Exception as e:
        return None, f"scikit-learn required: pip install scikit-learn ({e})"

    # One-hot encode y (groups)
    y_cat = pd.Categorical(y)
    Y = pd.get_dummies(y_cat, drop_first=False)

    # Drop columns with zero variance
    X_num = X.loc[:, X.var(axis=0) > 0]

    rows = []
    kf = KFold(n_splits=min(cv_folds, len(X_num)), shuffle=True, random_state=42)
    for n in range(1, min(max_comps, X_num.shape[1], len(X_num)) + 1):
        pls = PLSRegression(n_components=n)
        # R2Y on full fit (not CV)
        pls.fit(X_num, Y)
        Y_hat = pls.predict(X_num)
        r2y = r2_score(Y, Y_hat, multioutput='variance_weighted')
        # Q2 via CV: PRESS/TSS
        press = 0.0; tss = 0.0
        for train, test in kf.split(X_num):
            X_tr, X_te = X_num.iloc[train], X_num.iloc[test]
            Y_tr, Y_te = Y.iloc[train], Y.iloc[test]
            model = PLSRegression(n_components=n)
            model.fit(X_tr, Y_tr)
            Y_pred = model.predict(X_te)
            press += ((Y_te - Y_pred) ** 2).values.sum()
            tss += ((Y_te - Y_tr.mean()) ** 2).values.sum()
        q2 = 1.0 - (press / tss) if tss > 0 else np.nan
        rows.append(dict(ncomp=n, R2Y=r2y, Q2=q2))
    return pd.DataFrame(rows), None

# ---------- Loadings helpers (new) ----------

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out

def guess_axis_col(modality: str, df: pd.DataFrame) -> Optional[str]:
    """
    Try to find the x-axis column for the chosen modality.
    - Chromatography: retention time
    - NMR: ppm / chemical shift
    - IV/UV-Vis: wavelength (nm)
    - RAMAN: wavenumber (cm-1)
    """
    cols = [c.lower() for c in df.columns]
    cand_map = {
        "Chromatography": [r"^rt$", r"retention.*time", r"row retention time", r"^time$"],
        "NMR":            [r"^ppm$", r"chemical.*shift", r"^delta$", r"^shift$"],
        "IV":             [r"^wavelength", r"\(nm\)", r"\bnm\b", r"lambda"],
        "RAMAN":          [r"^wavenumber", r"\(cm-1\)", r"\bcm[- ]?1\b", r"raman.*shift"],
    }
    patterns = cand_map.get(modality, [])
    for pat in patterns:
        for i, c in enumerate(cols):
            if re.search(pat, c, re.I):
                return df.columns[i]
    # Fallback: if there is a single numeric column, use it
    numc = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numc) == 1:
        return numc[0]
    return None

def list_pc_columns(df: pd.DataFrame, kind: str) -> List[str]:
    """
    kind='PCA'  → PC1, PC2, ... (robust to 'PC1 (12.3%)')
    kind='PLS'  → Comp 1, Comp. 1, LV1, etc.
    """
    cols = []
    for c in df.columns:
        cl = str(c).strip().lower()
        if kind.upper() == "PCA":
            # PC1 or "PC1 (12.3%)"
            if re.match(r"^pc\s*\d+(\s*\(.*\))?$", cl):
                cols.append(c)
        else:
            # comp 1 / comp. 1 / lv1
            if re.match(r"^(comp\.?\s*\d+|lv\s*\d+)(\s*\(.*\))?$", cl):
                cols.append(c)
    # Sort by component index if possible
    def pc_num(name):
        m = re.search(r"(\d+)", str(name))
        return int(m.group(1)) if m else 999
    return sorted(cols, key=pc_num)

def stem_traces(x, y, name, color=None):
    """
    Build 'stem' (vertical line) traces for Plotly.
    """
    xs = []; ys = []
    for xi, yi in zip(x, y):
        xs += [xi, xi, None]
        ys += [0,  yi, None]
    return go.Scatter(x=xs, y=ys, mode="lines", name=name,
                      line=dict(width=1.5, color=color) if color else None)

def loadings_plot(df: pd.DataFrame, xcol: str, pc_cols: List[str],
                  style: str = "Lines", reverse_x: bool = False,
                  title: str = "", height: int = 600):
    """Return a Plotly Figure for loadings vs axis."""
    fig = go.Figure()
    # Sort by x for cleaner lines
    d = df[[xcol] + pc_cols].copy().dropna()
    d = d.sort_values(by=xcol)
    for i, pc in enumerate(pc_cols):
        if style == "Stems":
            fig.add_trace(stem_traces(d[xcol], d[pc], name=str(pc)))
        else:
            fig.add_scatter(x=d[xcol], y=d[pc], mode="lines", name=str(pc))
    fig.update_layout(
        template="simple_white", height=height,
        xaxis_title=xcol, yaxis_title="Loading", title=title, legend_title="Components"
    )
    if reverse_x:
        fig.update_xaxes(autorange="reversed")
    return fig


# ---- SAFE PREVIEW (NO PYARROW) ----

def show_df_safe(df: pd.DataFrame, rows: int = 20):
    # Always render as plain text to avoid pyarrow
    txt = df.head(rows).to_string(index=False)
    st.markdown(f"```text\n{txt}\n```")

def parse_metaboanalyst_xnorm(df: pd.DataFrame):
    """
    Parse MetaboAnalyst 'data_normalized.csv' where:
      - Columns = samples (plus a first column like 'Name')
      - A row with first cell 'Label' contains class labels for each sample column
      - Remaining rows = numeric features
    Returns: X (samples x features), y (Series indexed by sample)
    """
    d = df.copy()

    # Canonicalize col names and first column name
    d.columns = [str(c).strip() for c in d.columns]
    first_col = d.columns[0]

    # Find the 'Label' row (case-insensitive, first column)
    label_mask = d[first_col].astype(str).str.strip().str.lower().eq("label")
    if not label_mask.any():
        return None, None, "No 'Label' row found (first column should contain 'Label' in one row)."

    label_row_idx = d.index[label_mask][0]

    # Sample columns are every column except the first (e.g., 'Name')
    sample_cols = d.columns[1:]

    # y: classes from the Label row (align to sample columns)
    y = d.loc[label_row_idx, sample_cols].astype(str).str.strip()
    y.index = [str(c).strip() for c in sample_cols]

    # Drop the label row from the data
    d = d.drop(index=label_row_idx)

    # If first column is feature IDs (Name), set it as index and drop it from data
    d_features = d.drop(columns=[first_col])
    # Convert to numeric
    d_features = d_features.apply(pd.to_numeric, errors="coerce")

    # X: samples x features
    X = d_features.T
    # Make sure X rows (samples) match y index
    X.index = [str(ix).strip() for ix in X.index]
    y = y.reindex(X.index)

    # Drop samples with missing labels
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    # Drop all-NaN / zero-variance features
    X = X.loc[:, X.notna().any(axis=0)]
    X = X.loc[:, X.var(axis=0) > 0]

    if X.empty or y.empty:
        return None, None, "After cleaning, X or y is empty."
    return X, y, None


# ------------------------------
# --------- Sidebar IO ---------
# ------------------------------

st.sidebar.header("Inputs")
use_samples = st.sidebar.checkbox(
    "Use sample files from working dir",
    value=False,
    help=(
        "Looks for CSVs named pca_score.csv, plsda_score.csv, plsda_vip.csv in the current folder. "
        "(No metadata.csv required.)"
    ),
)

if use_samples:
    pca_score_file = "pca_score.csv"
    plsda_score_file = "plsda_score.csv"
    vip_file = "plsda_vip.csv"
    r2q2_file = "plsda_r2q2.csv"  # optional, if present
    xnorm_file = "data_normalized.csv"  # optional for computing R2/Q2

    def exists(p):
        return os.path.isfile(p)

    # Only require core files; others are optional
    if not all(map(exists, [pca_score_file, plsda_score_file, vip_file])):
        st.sidebar.error("Sample files not found in current directory. Upload files instead.")
        use_samples = False

if not use_samples:
    pca_score_upload = st.sidebar.file_uploader("pca_score.csv", type=["csv"])    
    pls_score_upload = st.sidebar.file_uploader("plsda_score.csv", type=["csv"])    
    vip_upload = st.sidebar.file_uploader("plsda_vip.csv", type=["csv"])    
    r2q2_upload = st.sidebar.file_uploader(
        "plsda_r2q2.csv (optional)", type=["csv"],
        help="R2/Q2 validation output from MetaboAnalyst (permutation or component table)"
    )
    xnorm_upload = st.sidebar.file_uploader(
        "data_normalized.csv (optional)", type=["csv"],
        help="Upload to compute R2/Q2 internally via CV if you don't have plsda_r2q2.csv."
    )

st.sidebar.subheader("Loadings (optional)")
pca_loadings_upload = st.sidebar.file_uploader(
    "pca_loadings.csv (from MetaboAnalyst)", type=["csv"],
    help="MetaboAnalyst → Download → pca_loadings.csv"
)
plsda_loadings_upload = st.sidebar.file_uploader(
    "plsda_loadings.csv (from MetaboAnalyst)", type=["csv"],
    help="MetaboAnalyst → Download → plsda_loadings.csv"
)

pca_loadings_df: Optional[pd.DataFrame] = None
plsda_loadings_df: Optional[pd.DataFrame] = None

st.sidebar.markdown("---")

# ---- Grouping options (no external metadata) ----
st.sidebar.markdown("---")
st.sidebar.subheader("Grouping (no metadata)")
mode = st.sidebar.radio(
    "How should we determine groups?",
    [
        "Use a column in the score CSV",
        "Parse from Sample via regex",
        "Single group (no coloring)",
    ],
    index=0,
)

regex_pat = st.sidebar.text_input(
    "Regex (if parsing from Sample)", value=r"^([^_]+)",
    help="Must include one capturing group to become the group label.")

# Optional color mapping (label:color)
color_text = st.sidebar.text_area(
    "Optional colors (label:hex or name, one per line)",
    value="",
)
user_palette = {}
for line in color_text.splitlines():
    if ":" in line:
        k, v = line.split(":", 1)
        user_palette[k.strip()] = v.strip()

# ------------------------------
# -------- Load the data -------
# ------------------------------
with st.spinner("Loading data…"):
    try:
        if use_samples:
            pca_score = read_csv_any(pca_score_file)
            plsda_score = read_csv_any(plsda_score_file)
            vip_df = read_csv_any(vip_file)
            r2q2_df = read_csv_any(r2q2_file) if 'r2q2_file' in locals() and os.path.isfile(r2q2_file) else None
            xnorm_df = read_csv_any(xnorm_file) if 'xnorm_file' in locals() and os.path.isfile(xnorm_file) else None

            # --- NEW: try sample loadings too (optional, won’t fail if absent)
            if os.path.isfile("pca_loadings.csv"):
                pca_loadings_df = read_csv_any("pca_loadings.csv")
            if os.path.isfile("plsda_loadings.csv"):
                plsda_loadings_df = read_csv_any("plsda_loadings.csv")

        else:
            if not (pca_score_upload and pls_score_upload and vip_upload):
                st.info("Upload pca_score.csv, plsda_score.csv and plsda_vip.csv or enable 'Use sample files'.")
                st.stop()

            pca_score = read_csv_any(pca_score_upload)
            plsda_score = read_csv_any(pls_score_upload)
            vip_df = read_csv_any(vip_upload)
            r2q2_df = read_csv_any(r2q2_upload) if 'r2q2_upload' in locals() and r2q2_upload is not None else None
            xnorm_df = read_csv_any(xnorm_upload) if 'xnorm_upload' in locals() and xnorm_upload is not None else None

            # --- NEW: read uploaded loadings if provided
            pca_loadings_df = read_csv_any(pca_loadings_upload) if pca_loadings_upload else None
            plsda_loadings_df = read_csv_any(plsda_loadings_upload) if plsda_loadings_upload else None

        # Normalize score files
        pca_score = ensure_sample_col(pca_score)
        plsda_score = ensure_sample_col(plsda_score)

        # Find axis columns
        pc1_col = try_guess_col(pca_score, ["pc1", "pc 1"]) or "PC1"
        pc2_col = try_guess_col(pca_score, ["pc2", "pc 2"]) or "PC2"

        c1 = try_guess_col(plsda_score, ["comp 1", "comp. 1", "comp1"]) or "Comp 1"
        c2 = try_guess_col(plsda_score, ["comp 2", "comp. 2", "comp2"]) or "Comp 2"

    except Exception as e:
        st.error(f"Failed to read data: {e}")
        st.stop()



# Determine group column from available options
candidate_group_cols = []
for cand in ["Group", "Class", "Classes", "ATTRIBUTE_Samples", "group", "class", "labels", "Label"]:
    if cand in pca_score.columns:
        candidate_group_cols.append(cand)

selected_group_col = None
if mode == "Use a column in the score CSV":
    if candidate_group_cols:
        selected_group_col = st.sidebar.selectbox("Pick group column", candidate_group_cols, index=0)
    else:
        st.sidebar.warning("No obvious group column found in pca_score.csv. Switch to 'Parse from Sample via regex'.")
        mode = "Parse from Sample via regex"

# Build working copies with a unified 'Group' column
pca_work = pca_score.copy()
pls_work = plsda_score.copy()

if mode == "Use a column in the score CSV" and selected_group_col:
    pca_work["Group"] = pca_work[selected_group_col].astype(str).str.strip()
    if selected_group_col in pls_work.columns:
        pls_work["Group"] = pls_work[selected_group_col].astype(str).str.strip()
    else:
        mapping = dict(zip(pca_work["Sample"], pca_work["Group"]))
        pls_work["Group"] = pls_work["Sample"].map(mapping).fillna("Group")

elif mode == "Parse from Sample via regex":
    pca_work["Group"] = build_group_from_regex(pca_work["Sample"], regex_pat)
    pls_work["Group"] = build_group_from_regex(pls_work["Sample"], regex_pat)

else:  # Single group
    pca_work["Group"] = "All"
    pls_work["Group"] = "All"

# >>> ADD THESE LINES HERE <<<
# Clean sample IDs for robust matching
pca_work["Sample"] = pca_work["Sample"].astype(str).str.strip()
pls_work["Sample"] = pls_work["Sample"].astype(str).str.strip()


# ------------------------------
# --------- Main Layout --------
# ------------------------------

st.title("MetaboAnalyst → Better Figures (no metadata)")
st.caption("PCA & PLS‑DA score plots with 90% confidence ellipses, plus VIP Top‑20.")

with st.expander("Peek at tables"):
    st.markdown("**PCA (first 20 rows):**")
    show_df_safe(pca_work, rows=20)
    st.markdown("**PLS‑DA (first 20 rows):**")
    show_df_safe(pls_work, rows=20)

# Tabs for PCA, PLS‑DA, VIP
pca_tab, pls_tab, load_tab = st.tabs(["PCA", "PLS-DA", "Full-resolution loadings plot"]) 

# ------------------------------
# -------------- PCA -----------
# ------------------------------
with pca_tab:
    st.subheader("PCA score plot")
    conf = st.slider("Confidence level", 0.50, 0.99, 0.90, 0.01)

    cats = pca_work["Group"].dropna().unique().tolist()
    if mode == "Single group (no coloring)":
        palette = {"All": px.colors.qualitative.Plotly[0]}
    else:
        palette = {c: user_palette.get(c, px.colors.qualitative.Plotly[i % 10]) for i, c in enumerate(cats)}

    fig = go.Figure()
    for g in cats:
        sub = pca_work[pca_work["Group"] == g]
        fig.add_scatter(
            x=sub[pc1_col], y=sub[pc2_col], mode="markers",
            name=g, legendgroup=g,
            marker=dict(size=10, line=dict(width=0.5, color="black")),
            marker_color=palette[g], hovertext=sub["Sample"], hoverinfo="text+x+y"
        )
        xy = ellipse_xy(sub[pc1_col], sub[pc2_col], conf=conf)
        if xy and mode != "Single group (no coloring)":
            xe, ye = xy
            fig.add_scatter(
                x=xe, y=ye, mode="lines", name=f"{g} ellipse", legendgroup=g,
                line=dict(width=2, color=palette[g]), hoverinfo="skip", showlegend=False
            )

    fig.update_layout(
        template="simple_white", height=600,
        xaxis_title=pc1_col, yaxis_title=pc2_col,
        legend_title="Group"
    )

    st.plotly_chart(fig, use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        st.download_button("Download HTML", data=fig.to_html(include_plotlyjs="cdn"), file_name="PCA_scorePlot.html")
    with colB:
        try:
            import kaleido  # noqa: F401
            png_bytes = fig.to_image(format="png", width=1600, height=1200, scale=2)
            st.download_button("Download PNG", data=png_bytes, file_name="PCA_scorePlot.png")
        except Exception:
            st.info("Install 'kaleido' to enable PNG export: pip install -U kaleido")

# ------------------------------
# ------------- PLS‑DA ---------
# ------------------------------
with pls_tab:
    st.subheader("PLS-DA: scores (top), VIP (middle), R2/Q2 validation (bottom)")
    conf_pls = st.slider("Confidence level", 0.50, 0.99, 0.90, 0.01, key="pls_conf")

    cats_pls = pls_work["Group"].dropna().unique().tolist()
    if mode == "Single group (no coloring)":
        palette_pls = {"All": px.colors.qualitative.Plotly[0]}
    else:
        palette_pls = {c: user_palette.get(c, px.colors.qualitative.Plotly[i % 10]) for i, c in enumerate(cats_pls)}

    # ===== TOP: PLS-DA SCORE PLOT =====
    fig2 = go.Figure()
    for g in cats_pls:
        sub = pls_work[pls_work["Group"] == g]
        fig2.add_scatter(
            x=sub[c1], y=sub[c2], mode="markers",
            name=g, legendgroup=g,
            marker=dict(size=10, line=dict(width=0.5, color="black")),
            marker_color=palette_pls[g], hovertext=sub["Sample"], hoverinfo="text+x+y"
        )
        xy2 = ellipse_xy(sub[c1], sub[c2], conf=conf_pls)
        if xy2 and mode != "Single group (no coloring)":
            xe, ye = xy2
            fig2.add_scatter(
                x=xe, y=ye, mode="lines", name=f"{g} ellipse", legendgroup=g,
                line=dict(width=2, color=palette_pls[g]), hoverinfo="skip", showlegend=False
            )

    fig2.update_layout(
        template="simple_white", height=600,
        xaxis_title=c1, yaxis_title=c2,
        legend_title="Group"
    )

    st.plotly_chart(fig2, use_container_width=True)

    cA, cB = st.columns(2)
    with cA:
        st.download_button("Download HTML (scores)", data=fig2.to_html(include_plotlyjs="cdn"),
                           file_name="PLSDA_scorePlot.html")
    with cB:
        try:
            import kaleido  # noqa: F401
            png2 = fig2.to_image(format="png", width=1600, height=1200, scale=2)
            st.download_button("Download PNG (scores)", data=png2, file_name="PLSDA_scorePlot.png")
        except Exception:
            st.info("Install 'kaleido' to enable PNG export: pip install -U kaleido")

    # ===== MIDDLE: VIP TOP-N =====
    st.markdown("---")
    st.markdown("### VIP Top-N (Component 1)")

    vip_col = try_guess_col(vip_df, ["comp. 1", "comp 1", "vip 1", "vip comp 1"]) or "Comp. 1"
    name_col = "Unnamed: 0" if "Unnamed: 0" in vip_df.columns else vip_df.columns[0]

    # How many features to show
    n_available = int(vip_df[vip_col].notna().sum())
    if n_available == 0:
        st.warning(f"No non-NA values found in '{vip_col}' of plsda_vip.csv.")
    else:
        n_default = 20 if n_available >= 20 else n_available
        n_min = 5 if n_available >= 5 else 1
        n_vip = st.slider("Number of VIP features", min_value=n_min, max_value=n_available,
                          value=n_default, step=1)

        topN = (
            vip_df.dropna(subset=[vip_col])
                  .nlargest(n_vip, vip_col)
                  .copy()
        )
        topN[name_col] = topN[name_col].astype(str)

        fig3 = px.bar(
            topN.sort_values(vip_col, ascending=True),
            x=vip_col, y=name_col, orientation="h", template="simple_white",
            title=f"Top {n_vip} by {vip_col}"
        )
        fig3.update_layout(height=600, yaxis=dict(title="Feature"), xaxis=dict(title=vip_col))
        st.plotly_chart(fig3, use_container_width=True)

        vA, vB = st.columns(2)
        with vA:
            st.download_button(
                "Download HTML (VIPs)",
                data=fig3.to_html(include_plotlyjs="cdn"),
                file_name=f"PLSDA_VIPs_top{n_vip}.html"
            )
        with vB:
            try:
                import kaleido  # noqa: F401
                png3 = fig3.to_image(format="png", width=1200, height=1200, scale=2)
                st.download_button(
                    "Download PNG (VIPs)",
                    data=png3,
                    file_name=f"PLSDA_VIPs_top{n_vip}.png"
                )
            except Exception:
                st.info("Install 'kaleido' to enable PNG export: pip install -U kaleido")

# -------------------------------------------------------
# -------------- Optional Loading plot as a line --------
# -------------------------------------------------------
with load_tab:
    st.subheader("Full-resolution loadings plot")
    st.caption("Upload *pca_loadings.csv* and/or *plsda_loadings.csv* from MetaboAnalyst and pick your modality.")

    modality = st.radio("Modality (affects x-axis detection & defaults)",
                        ["Chromatography", "NMR", "IV", "RAMAN"], index=0, horizontal=True)
    src_kind = st.radio("Source", ["PCA loadings", "PLS-DA loadings"], index=0, horizontal=True)

    # --- SAFE: don’t stop the whole app if missing
    if src_kind == "PCA loadings":
        if pca_loadings_df is None or pca_loadings_df.empty:
            st.warning("Upload **pca_loadings.csv** in the sidebar to use this view.")
            st.stop()  # <- If you prefer to keep the stop, it’s ok here since we are in the last tab block.
        df_load = _norm_cols(pca_loadings_df)
    else:
        if plsda_loadings_df is None or plsda_loadings_df.empty:
            st.warning("Upload **plsda_loadings.csv** in the sidebar to use this view.")
            st.stop()
        df_load = _norm_cols(plsda_loadings_df)

    default_x = guess_axis_col(modality, df_load)
    numeric_cols = df_load.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = list(df_load.columns)
    x_index = all_cols.index(default_x) if default_x in all_cols else 0
    xcol = st.selectbox("X-axis column", options=all_cols, index=x_index)

    if xcol not in numeric_cols:
        st.info("Selected x-axis is not numeric; trying to coerce.")
        df_load[xcol] = pd.to_numeric(df_load[xcol], errors="coerce")
        if not np.issubdtype(df_load[xcol].dtype, np.number):
            st.error("X-axis column must be numeric after coercion.")
            st.stop()

    comp_cols = list_pc_columns(df_load, "PCA" if src_kind.startswith("PCA") else "PLS")
    if not comp_cols:
        st.error("Could not find component columns (e.g., PC1/PC2 or Comp 1/Comp 2).")
        st.stop()

    # Detect component columns
    comp_cols = list_pc_columns(df_load, "PCA" if src_kind.startswith("PCA") else "PLS")
    if not comp_cols:
        st.error("Could not find component columns (e.g., PC1/PC2 or Comp 1/Comp 2).")
        st.stop()

    sel_pcs = st.multiselect(
        "Select components to plot", comp_cols,
        default=comp_cols[:2] if len(comp_cols) >= 2 else comp_cols
    )
    if not sel_pcs:
        st.info("Pick at least one component.")
        st.stop()

    # Style & options
    col1, col2, col3 = st.columns(3)
    with col1:
        style = st.selectbox("Style", ["Lines", "Stems"], index=0,
                             help="Stems = vertical stick plot")
    with col2:
        reverse_x = st.checkbox("Reverse x-axis (NMR ppm)", value=(modality == "NMR"))
    with col3:
        fig_h = st.number_input("Figure height (px)", min_value=400, max_value=2000, value=700, step=50)

    title = f"{src_kind}: {' / '.join(sel_pcs)} — {modality}"
    fig = loadings_plot(df_load, xcol, sel_pcs, style=style, reverse_x=reverse_x, title=title, height=fig_h)
    st.plotly_chart(fig, use_container_width=True)

    # Exports
    st.markdown("**Export**")
    e1, e2, e3 = st.columns([1,1,2])
    with e1:
        st.download_button("Download HTML", data=fig.to_html(include_plotlyjs="cdn"),
                           file_name=f"{src_kind.replace(' ', '_')}_{modality}_loadings.html")
    with e2:
        try:
            import kaleido  # noqa
            png_w = st.number_input("PNG width (px)", min_value=800, max_value=4000, value=2000, step=100, key="load_pngw")
            png_h = st.number_input("PNG height (px)", min_value=600, max_value=3000, value=fig_h, step=50, key="load_pngh")
            scale = st.slider("Scale", 1, 5, 2)
            png = fig.to_image(format="png", width=int(png_w), height=int(png_h), scale=int(scale))
            st.download_button("Download PNG", data=png,
                               file_name=f"{src_kind.replace(' ', '_')}_{modality}_loadings.png")
        except Exception:
            st.info("Install **kaleido** to enable PNG export: `pip install -U kaleido`")


    # ===== BOTTOM: R2/Q2 VALIDATION =====
    st.markdown("---")
    st.markdown("### R2/Q2 validation")

    with st.expander("R2/Q2 diagnostics"):
        has_r2q2 = ('r2q2_df' in locals()) and (r2q2_df is not None) and (not r2q2_df.empty)
        has_xnorm = ('xnorm_df' in locals()) and (xnorm_df is not None) and (not xnorm_df.empty)

        diag = pd.DataFrame({
            "has_plsda_r2q2_csv": [has_r2q2],
            "has_data_normalized_csv": [has_xnorm],
            "n_samples_PLS": [len(pls_work)],
        })
        st.markdown("**Inputs**")
        st.markdown(f"```text\n{diag.to_string(index=False)}\n```")

        st.markdown("**First 5 PLS sample IDs**")
        st.markdown("```text\n" + "\n".join(pls_work["Sample"].astype(str).head(5).tolist()) + "\n```")

        if has_xnorm:
            st.markdown(f"**data_normalized.csv shape:** `{xnorm_df.shape}`")
            st.markdown("**Head (2 rows)**")
            st.markdown(f"```text\n{xnorm_df.head(2).to_string(index=False)}\n```")



    # Option A: if user supplied plsda_r2q2.csv, plot it (already handled above)
    external_has_plot = False
    if 'r2q2_df' in locals() and r2q2_df is not None and not r2q2_df.empty:
        try:
            def pick_r2q2_columns(df: pd.DataFrame):
                x_candidates = [c for c in df.columns if any(k in str(c).lower() for k in ["perm", "component", "ncomp", "comp", "index"]) ]
                xcol = x_candidates[0] if x_candidates else df.columns[0]
                r2_candidates = [c for c in df.columns if "r2" in str(c).lower()]
                q2_candidates = [c for c in df.columns if "q2" in str(c).lower()]
                r2col = r2_candidates[0] if r2_candidates else None
                q2col = q2_candidates[0] if q2_candidates else None
                return xcol, r2col, q2col

            xcol, r2col, q2col = pick_r2q2_columns(r2q2_df)
            fig4 = go.Figure()
            if r2col is not None:
                fig4.add_scatter(x=r2q2_df[xcol], y=r2q2_df[r2col], mode="lines+markers",
                                 name=str(r2col), line=dict(width=2))
            if q2col is not None:
                fig4.add_scatter(x=r2q2_df[xcol], y=r2q2_df[q2col], mode="lines+markers",
                                 name=str(q2col), line=dict(width=2))
            try:
                xmin, xmax = r2q2_df[xcol].min(), r2q2_df[xcol].max()
                fig4.add_shape(type="line", x0=xmin, x1=xmax, y0=0, y1=0,
                               line=dict(width=1, dash="dash", color="black"))
            except Exception:
                pass
            fig4.update_layout(template="simple_white", height=450,
                               xaxis_title=str(xcol), yaxis_title="Score",
                               yaxis=dict(range=[-0.5, 1.05]))
            st.plotly_chart(fig4, use_container_width=True)
            external_has_plot = True

            rA, rB = st.columns(2)
            with rA:
                st.download_button("Download HTML (R2Q2)", data=fig4.to_html(include_plotlyjs="cdn"),
                                   file_name="PLSDA_R2Q2.html")
            with rB:
                try:
                    import kaleido  # noqa: F401
                    png4 = fig4.to_image(format="png", width=1400, height=900, scale=2)
                    st.download_button("Download PNG (R2Q2)", data=png4, file_name="PLSDA_R2Q2.png")
                except Exception:
                    st.info("Install 'kaleido' to enable PNG export: pip install -U kaleido")
        except Exception as e:
            st.warning(f"Could not render supplied R2/Q2 file: {e}")

    # Option B: compute internally from data_normalized.csv and Group labels
    if not external_has_plot:
        st.markdown("#### Compute R2/Q2 from data_normalized.csv (cross-validation)")
        if 'xnorm_df' in locals() and xnorm_df is not None and not xnorm_df.empty:
            # >>> Parse X and y directly from the normalized file (uses the 'Label' row) <<<
            X, y, msg = parse_metaboanalyst_xnorm(xnorm_df)
            if msg:
                st.warning(f"Could not parse data_normalized.csv: {msg}")
            else:
                # Diagnostics
                st.markdown("**Compute R2/Q2 diagnostics**")
                st.markdown(f"- X shape: **{X.shape}** (rows=samples, cols=features)")
                #st.markdown(f"- # groups: **{len(pd.unique(y))}** — {sorted(map(str, pd.unique(y)))}")
                st.markdown(f"- Zero-variance features dropped: **{int((X.var(axis=0) == 0).sum())}**")

                # Controls
                max_comps = st.slider("Max # of components", 1, min(10, X.shape[1], len(X)), 5)
                folds = st.slider("CV folds", 3, min(10, len(X)), 7)

                # Compute and plot
                res_df, err = compute_pls_r2q2(X, y, max_comps=max_comps, cv_folds=folds)
                if err:
                    st.error(err)
                elif res_df is None or res_df.empty:
                    st.warning("Failed to compute R2/Q2.")
                else:
                    res_df = res_df.copy()
                    res_df = res_df[~(res_df["R2Y"].isna() & res_df["Q2"].isna())]
                    if res_df.empty:
                        st.warning("All R2/Q2 values are NaN after cleaning — check your labels and data.")
                    else:
                        yvals = res_df[["R2Y", "Q2"]].to_numpy().astype(float)
                        finite_vals = yvals[np.isfinite(yvals)]
                        y_min = float(np.min(finite_vals)) if finite_vals.size else -0.05
                        y_min = min(-0.05, y_min - 0.05)
                        y_max = 1.05

                        fig5 = go.Figure()
                        if res_df["R2Y"].notna().any():
                            fig5.add_bar(x=res_df['ncomp'], y=res_df['R2Y'], name='R2Y')
                        if res_df["Q2"].notna().any():
                            fig5.add_bar(x=res_df['ncomp'], y=res_df['Q2'], name='Q2')
                        fig5.update_layout(
                            template='simple_white', barmode='group', height=450,
                            xaxis_title='Number of components', yaxis_title='Score',
                            yaxis=dict(range=[y_min, y_max])
                        )
                        st.plotly_chart(fig5, use_container_width=True)

                        gA, gB = st.columns(2)
                        with gA:
                            st.download_button("Download HTML (computed R2/Q2)",
                                               data=fig5.to_html(include_plotlyjs="cdn"),
                                               file_name="PLSDA_R2Q2_computed.html")
                        with gB:
                            try:
                                import kaleido  # noqa: F401
                                png5 = fig5.to_image(format='png', width=1400, height=900, scale=2)
                                st.download_button("Download PNG (computed R2/Q2)", data=png5,
                                                   file_name="PLSDA_R2Q2_computed.png")
                            except Exception:
                                st.info("Install 'kaleido' to enable PNG export: pip install -U kaleido")
        else:
            st.info("Upload 'data_normalized.csv' in the sidebar to compute R2/Q2 via cross-validation (no pyarrow required).")

# ------------------------------
# -------------- Footer --------
# ------------------------------

st.markdown(
    """
    ---
    **Tips**
    - First things first: run your study, process your data, analyze it using MetaboAnalyst, download the .zip folder and extract into a target folder.
    - This app does *not* require metadata.csv. Choose an existing group column or parse from Sample with a regex.
    - Axis labels are detected automatically even if MetaboAnalyst writes `PC1 (12.3%)` or `Comp. 1`/`Comp 1`.
    - Use the color mapping box in the sidebar to pin specific colors to classes.
    - Increase confidence to 0.95/0.99 if you want larger ellipses.
    """
)


