import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.units import inch
import io
import re
import warnings
warnings.filterwarnings('ignore')

def nc(series):
    """Safely coerce a pandas Series to numeric — returns 0 for non-numeric/NaN values."""
    return pd.to_numeric(series, errors='coerce').fillna(0)

# ════════════════════════════════════════════════════════════════════════════
# VELYTICS SMART CLEAN ENGINE — Auto-fixes 23 types of Excel mess
# ════════════════════════════════════════════════════════════════════════════

def smart_clean(df_raw, filename=""):
    """
    Master cleaning function. Returns (df_clean, fixes_log)
    fixes_log = list of strings describing what was fixed.
    """
    fixes = []
    df = df_raw.copy()

    # ── 1. DETECT REAL HEADER ROW ────────────────────────────────────────────
    # Scan first 10 rows — find row with most non-null, non-numeric text values
    def header_score(row):
        score = 0
        for val in row:
            v = str(val).strip()
            if v and v.lower() not in ['nan','none','']:
                if not re.match(r'^[\d,.\-\+\(\)₹$%\s]+$', v):
                    score += 1
        return score

    best_row = 0
    best_score = header_score(df.iloc[0])
    for i in range(1, min(10, len(df))):
        s = header_score(df.iloc[i])
        if s > best_score:
            best_score = s
            best_row = i

    # Always assign headers from best_row (0 = normal file, >0 = file with title rows above)
    df.columns = df.iloc[best_row].astype(str).str.strip()
    df = df.iloc[best_row + 1:].reset_index(drop=True)
    if best_row > 0:
        fixes.append(f"📋 Detected real headers on row {best_row + 1} — skipped {best_row} title row(s)")

    # ── 2. CLEAN COLUMN NAMES ────────────────────────────────────────────────
    original_cols = list(df.columns)
    new_cols = []
    seen = {}
    for col in df.columns:
        c = str(col).strip()
        c = re.sub(r'\s+', ' ', c)           # collapse spaces
        c = c.strip()
        if c.lower() in ['nan','none','unnamed','']:
            c = f"Column_{len(new_cols)+1}"
        # deduplicate
        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
        new_cols.append(c)
    df.columns = new_cols
    renamed = sum(1 for a,b in zip(original_cols, new_cols) if str(a).strip() != b)
    if renamed > 0:
        fixes.append(f"🏷️ Cleaned {renamed} column name(s) — removed extra spaces and unnamed columns")

    # ── 3. REMOVE BLANK ROWS & COLUMNS ──────────────────────────────────────
    before_rows = len(df)
    df.dropna(how='all', inplace=True)
    blank_rows = before_rows - len(df)
    if blank_rows > 0:
        fixes.append(f"🗑️ Removed {blank_rows} completely blank row(s)")

    before_cols = len(df.columns)
    df.dropna(axis=1, how='all', inplace=True)
    blank_cols = before_cols - len(df.columns)
    if blank_cols > 0:
        fixes.append(f"🗑️ Removed {blank_cols} completely blank column(s)")

    # ── 4. REMOVE TOTAL / SUMMARY ROWS ──────────────────────────────────────
    total_keywords = ['total','grand total','subtotal','sum','average','avg','overall']
    mask_total = df.apply(lambda row: any(
        str(v).lower().strip() in total_keywords for v in row
    ), axis=1)
    total_removed = mask_total.sum()
    if total_removed > 0:
        df = df[~mask_total].reset_index(drop=True)
        fixes.append(f"🗑️ Removed {total_removed} summary/total row(s) mixed in data")

    # ── 5. REMOVE DUPLICATE HEADER ROWS ─────────────────────────────────────
    col_names_set = set(str(c).lower().strip() for c in df.columns)
    dup_header_mask = df.apply(lambda row: sum(
        str(v).lower().strip() in col_names_set for v in row
    ) >= len(df.columns) * 0.6, axis=1)
    dup_headers = dup_header_mask.sum()
    if dup_headers > 0:
        df = df[~dup_header_mask].reset_index(drop=True)
        fixes.append(f"📋 Removed {dup_headers} repeated header row(s) found mid-sheet")

    # ── 6. REMOVE TEST / GARBAGE ROWS ───────────────────────────────────────
    garbage_patterns = [r'^(test|xxx|asdf|qwerty|dummy|sample|n/a|na|nil|tbd|temp|delete)$']
    def is_garbage(row):
        non_null = [str(v).strip().lower() for v in row if str(v).strip().lower() not in ['','nan','none']]
        if not non_null:
            return True
        garbage = sum(1 for v in non_null if any(re.match(p, v) for p in garbage_patterns))
        return garbage >= len(non_null) * 0.6
    garbage_mask = df.apply(is_garbage, axis=1)
    garbage_removed = garbage_mask.sum()
    if garbage_removed > 0:
        df = df[~garbage_mask].reset_index(drop=True)
        fixes.append(f"🗑️ Removed {garbage_removed} test/garbage row(s)")

    # ── 7. REMOVE DUPLICATE ROWS ────────────────────────────────────────────
    before = len(df)
    # For large files use subset of key columns to speed up dedup
    if len(df) > 100_000:
        key_cols = df.columns[:min(5, len(df.columns))].tolist()
        df.drop_duplicates(subset=key_cols, inplace=True)
    else:
        df.drop_duplicates(inplace=True)
    dups = before - len(df)
    if dups > 0:
        fixes.append(f"🔁 Removed {dups} duplicate row(s)")
    df.reset_index(drop=True, inplace=True)

    # ── LARGE FILE OPTIMISATION ──────────────────────────────────────────
    # Downcast numeric columns only — no category conversion (causes pyarrow issues)
    if len(df) > 50_000:
        for col in df.select_dtypes(include=['float64']).columns:
            try:
                df[col] = pd.to_numeric(df[col], downcast='float')
            except Exception:
                pass
        for col in df.select_dtypes(include=['int64']).columns:
            try:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            except Exception:
                pass

    # ── 8. CLEAN TEXT COLUMNS ────────────────────────────────────────────────
    text_fixes = 0
    for col in df.select_dtypes(include='object').columns:
        try:
            original_str = df[col].astype(str).fillna('')
            # Strip whitespace
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            # Fix 'nan' strings back to NaN
            df[col] = df[col].replace({'nan': np.nan, 'none': np.nan, 'None': np.nan,
                                        'NULL': np.nan, 'null': np.nan, 'N/A': np.nan,
                                        'n/a': np.nan, 'NA': np.nan, '': np.nan})
            # Title case for category-like columns (low unique count)
            nuniq = df[col].nunique()
            if 0 < nuniq < 30:
                df[col] = df[col].str.title()
            # Count changes safely
            new_str = df[col].astype(str).fillna('')
            changed = int((new_str != original_str).sum())
            text_fixes += changed
        except Exception:
            pass
    if text_fixes > 0:
        fixes.append(f"✍️ Fixed {text_fixes} text value(s) — standardised case, removed extra spaces")

    # ── 9. DETECT & CONVERT NUMBER COLUMNS ──────────────────────────────────
    num_fixes = 0
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(50)
            # Check if looks like numbers with symbols
            cleaned = sample.astype(str).str.replace(r'[₹$€£,\s%\(\)]', '', regex=True)\
                                         .str.replace(r'^\((.+)\)$', r'-\1', regex=True)
            numeric_count = nc(cleaned).notna().sum()
            if numeric_count >= len(sample) * 0.7 and len(sample) > 0:
                df[col] = df[col].astype(str)\
                    .str.replace(r'[₹$€£,\s%]', '', regex=True)\
                    .str.replace(r'^\((.+)\)$', r'-\1', regex=True)
                converted = nc(df[col])
                success = converted.notna().sum()
                if success > 0:
                    df[col] = converted
                    num_fixes += 1
    if num_fixes > 0:
        fixes.append(f"🔢 Converted {num_fixes} column(s) from text-numbers to numeric (removed ₹,$,commas,%)")

    # ── 10. DETECT & PARSE DATE COLUMNS ─────────────────────────────────────
    date_fixes = 0
    date_formats = ['%d/%m/%Y','%m/%d/%Y','%Y-%m-%d','%d-%m-%Y','%d/%m/%y',
                    '%m/%d/%y','%b-%y','%B %Y','%Y/%m/%d','%d %b %Y','%d-%b-%Y']
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(30).astype(str)
            date_count = 0
            for val in sample:
                if pd.to_datetime(val, errors='coerce', dayfirst=True) is not pd.NaT:
                    try:
                        result = pd.to_datetime(val, errors='coerce', dayfirst=True)
                        if result is not pd.NaT and str(result) != 'NaT':
                            date_count += 1
                    except:
                        pass
            if date_count >= len(sample) * 0.6 and len(sample) > 0:
                converted = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
                if converted.notna().sum() >= len(df) * 0.5:
                    df[col] = converted
                    date_fixes += 1
        elif df[col].dtype in ['int64','float64']:
            # Excel serial date detection (values between 30000-50000 range)
            sample_vals = df[col].dropna()
            if len(sample_vals) > 0 and sample_vals.between(30000, 50000).mean() > 0.7:
                try:
                    df[col] = pd.to_datetime(df[col], unit='D', origin='1899-12-30', errors='coerce')
                    date_fixes += 1
                    fixes.append(f"📅 Converted '{col}' from Excel serial numbers to real dates")
                except:
                    pass
    if date_fixes > 0:
        fixes.append(f"📅 Parsed {date_fixes} date column(s) — standardised mixed date formats")

    # ── 11. FILL MISSING NUMERIC VALUES ─────────────────────────────────────
    missing_filled = 0
    for col in df.select_dtypes(include='number').columns:
        miss = df[col].isnull().sum()
        if miss > 0:
            df[col] = df[col].fillna(df[col].median())
            missing_filled += miss
    if missing_filled > 0:
        fixes.append(f"🔧 Filled {missing_filled} missing numeric value(s) with column median")

    # ── 12. DETECT PERCENTAGE COLUMNS ───────────────────────────────────────
    for col in df.select_dtypes(include='number').columns:
        vals = df[col].dropna()
        if len(vals) > 0 and vals.between(0, 100).mean() > 0.9 and 'pct' in col.lower() or '%' in col:
            pass  # already fine as 0-100 scale

    df.reset_index(drop=True, inplace=True)
    return df, fixes


CHUNK_SIZE = 100_000   # rows per chunk for large files
LARGE_FILE_THRESHOLD = 50_000  # rows — show progress bar above this

def load_file(uploaded_file):
    """
    Load any Excel/CSV with large-file support.
    - CSV files > 50K rows: chunked loading with progress bar
    - Excel files: multi-sheet selector, optimised dtypes
    - Also supports JSON files
    """
    name = uploaded_file.name.lower()

    # ── JSON ─────────────────────────────────────────────────────────────
    if name.endswith('.json'):
        return pd.read_json(uploaded_file)

    # ── CSV ──────────────────────────────────────────────────────────────
    if name.endswith('.csv'):
        # Peek at file size
        uploaded_file.seek(0, 2)
        file_size = uploaded_file.tell()
        uploaded_file.seek(0)

        # Large CSV: chunk it
        if file_size > 10 * 1024 * 1024:  # > 10MB
            progress_bar = st.progress(0, text="⚡ Loading large file...")
            chunks = []
            try:
                chunk_iter = pd.read_csv(uploaded_file, chunksize=CHUNK_SIZE, low_memory=False)
            except Exception:
                uploaded_file.seek(0)
                chunk_iter = pd.read_csv(uploaded_file, chunksize=CHUNK_SIZE, encoding='latin-1', low_memory=False)

            total_rows = 0
            for i, chunk in enumerate(chunk_iter):
                chunks.append(chunk)
                total_rows += len(chunk)
                pct = min(0.95, (i+1) * 0.1)
                progress_bar.progress(pct, text=f"⚡ Loaded {total_rows:,} rows...")

            progress_bar.progress(1.0, text=f"✅ {total_rows:,} rows loaded!")
            progress_bar.empty()
            return pd.concat(chunks, ignore_index=True)

        # Small CSV: direct load
        try:
            return pd.read_csv(uploaded_file, low_memory=False)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding='latin-1', low_memory=False)

    # ── EXCEL ─────────────────────────────────────────────────────────────
    xl = pd.ExcelFile(uploaded_file)
    sheets = xl.sheet_names

    if len(sheets) > 1:
        if 'selected_sheet' not in st.session_state:
            st.session_state.selected_sheet = sheets[0]
        selected = st.sidebar.selectbox("📄 Excel Sheet", sheets, key="sheet_selector")
        st.session_state.selected_sheet = selected
        sheet_name = selected
    else:
        sheet_name = sheets[0]

    # Check row count first without loading data
    df_peek = pd.read_excel(xl, sheet_name=sheet_name, header=None, nrows=1)
    nrows_estimate = None
    try:
        import openpyxl
        wb = openpyxl.load_workbook(uploaded_file, read_only=True, data_only=True)
        ws = wb[sheet_name if isinstance(sheet_name, str) else xl.sheet_names[sheet_name]]
        nrows_estimate = ws.max_row
        wb.close()
        uploaded_file.seek(0)
    except Exception:
        pass

    if nrows_estimate and nrows_estimate > LARGE_FILE_THRESHOLD:
        progress_bar = st.progress(0, text=f"⚡ Loading {nrows_estimate:,} rows from Excel...")
        df = pd.read_excel(xl, sheet_name=sheet_name, header=None)
        progress_bar.progress(1.0, text=f"✅ {len(df):,} rows loaded!")
        progress_bar.empty()
        return df

    return pd.read_excel(xl, sheet_name=sheet_name, header=None)


def show_clean_report(fixes, df_clean, df_raw):
    """Show a beautiful cleaning report."""
    if not fixes:
        st.success(f"✅ File loaded — {len(df_clean):,} rows · {len(df_clean.columns)} columns · Already clean!")
        return
    total_fixes = len(fixes)
    with st.expander(f"✅ File loaded & auto-cleaned — {len(df_clean):,} rows · {len(df_clean.columns)} columns · **{total_fixes} issues fixed** — click to see details", expanded=False):
        cols = st.columns(3)
        cols[0].metric("Rows Before", f"{len(df_raw):,}")
        cols[1].metric("Rows After", f"{len(df_clean):,}")
        cols[2].metric("Issues Fixed", total_fixes)
        st.markdown("---")
        for fix in fixes:
            st.markdown(f"<div style='background:#f0fdf4;border-left:3px solid #10b981;padding:8px 12px;border-radius:6px;margin:4px 0;font-size:0.82rem;color:#166534;'>{fix}</div>", unsafe_allow_html=True)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Velytics — Your data scientist. No hiring required.",
    page_icon="⚡", layout="wide"
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.main { background: #f8fafc; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5e7eb; }

/* Hero */
.hero { background: #0a0a0a; border-radius: 16px; padding: 36px 40px; margin-bottom: 28px; display:flex; justify-content:space-between; align-items:center; }
.hero-left h1 { color: white; font-size: 2.2rem; font-weight: 900; letter-spacing: -1.5px; margin: 0 0 6px 0; }
.hero-left h1 span { color: #818cf8; }
.hero-left p { color: #9ca3af; font-size: 0.95rem; margin: 0; }
.hero-right { display: flex; gap: 24px; }
.hero-stat { text-align: center; }
.hero-stat-num { color: white; font-size: 1.6rem; font-weight: 800; }
.hero-stat-label { color: #6b7280; font-size: 0.7rem; text-transform: uppercase; font-weight: 500; }

/* Module selector — radio styled as pills */
div[data-testid="stRadio"] > div { display: flex; flex-wrap: wrap; gap: 8px; }
div[data-testid="stRadio"] label {
    display: inline-flex !important; align-items: center; gap: 6px;
    padding: 9px 18px !important; border-radius: 100px !important;
    border: 1.5px solid #e5e7eb !important; background: white !important;
    cursor: pointer; font-size: 0.83rem !important; font-weight: 600 !important;
    color: #374151 !important; transition: all 0.15s; white-space: nowrap;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
div[data-testid="stRadio"] label:hover { border-color: #6366f1 !important; color: #6366f1 !important; background: #fafafa !important; }
div[data-testid="stRadio"] label[data-checked="true"],
div[data-testid="stRadio"] label[aria-checked="true"] {
    background: #6366f1 !important; color: white !important;
    border-color: #6366f1 !important; box-shadow: 0 2px 8px rgba(99,102,241,0.35) !important;
}
div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { font-size: 0.83rem !important; font-weight: 600 !important; }
div[data-testid="stRadio"] input[type="radio"] { display: none !important; }
div[data-testid="stRadio"] > label { display: none; }

/* Feature pills */
.fpill { display:inline-block; background:#f1f5f9; color:#475569; border-radius:100px;
    padding:4px 12px; font-size:0.75rem; font-weight:600; margin:3px; }

/* Upload */
.upload-hint { text-align:center; padding:40px 20px; background:white; border-radius:16px;
    border: 2px dashed #e5e7eb; margin-top:8px; }
.upload-hint h3 { font-size:1.3rem; font-weight:800; color:#0a0a0a; margin:12px 0 6px; }
.upload-hint p { color:#6b7280; font-size:0.9rem; margin:0; }

/* Section header */
.sh { display:flex; align-items:center; gap:10px; margin:28px 0 14px;
    padding-bottom: 10px; border-bottom: 2px solid #f1f5f9; }
.sh-icon { font-size:1.2rem; }
.sh-title { font-size:1rem; font-weight:800; color:#0a0a0a; letter-spacing:-0.3px; }
.sh-badge { background:#eef2ff; color:#4f46e5; font-size:0.7rem; font-weight:600;
    padding:2px 8px; border-radius:100px; }

/* KPI cards */
.kpi-grid { display:grid; gap:12px; margin-bottom:20px; }
.kpi-card { background:white; border-radius:12px; padding:18px 20px;
    box-shadow:0 1px 4px rgba(0,0,0,0.05); border-top: 3px solid #6366f1; }
.kpi-card.red { border-top-color: #ef4444; }
.kpi-card.amber { border-top-color: #f59e0b; }
.kpi-card.green { border-top-color: #10b981; }
.kpi-card.purple { border-top-color: #8b5cf6; }
.kpi-num { font-size:1.6rem; font-weight:900; color:#0a0a0a; letter-spacing:-1px; }
.kpi-label { font-size:0.72rem; color:#6b7280; font-weight:600; text-transform:uppercase; margin-top:4px; }
.kpi-sub { font-size:0.75rem; margin-top:6px; font-weight:500; }

/* Alert / insight boxes */
.abox { border-radius:10px; padding:13px 16px; margin:8px 0; font-size:0.875rem; line-height:1.6; }
.abox.red   { background:#fef2f2; border-left:4px solid #ef4444; color:#991b1b; }
.abox.amber { background:#fffbeb; border-left:4px solid #f59e0b; color:#92400e; }
.abox.green { background:#f0fdf4; border-left:4px solid #10b981; color:#166534; }
.abox.blue  { background:#eff6ff; border-left:4px solid #3b82f6; color:#1e40af; }

/* Tabs override */
[data-baseweb="tab-list"] { gap: 4px; background: #f1f5f9; border-radius: 10px; padding: 4px; }
[data-baseweb="tab"] { border-radius: 8px !important; font-weight: 600 !important; font-size: 0.82rem !important; }
[aria-selected="true"] { background: white !important; box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important; }

div[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 800 !important; }
</style>
""", unsafe_allow_html=True)

# ── HERO BANNER ──────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-left'>
        <h1>Vely<span>tics</span></h1>
        <p>Your data scientist. No hiring required.</p>
    </div>
    <div class='hero-right'>
        <div class='hero-stat'><div class='hero-stat-num'>20</div><div class='hero-stat-label'>Industries</div></div>
        <div class='hero-stat'><div class='hero-stat-num'>150+</div><div class='hero-stat-label'>Analyses</div></div>
        <div class='hero-stat'><div class='hero-stat-num'>60s</div><div class='hero-stat-label'>Time to insight</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── MODULE SELECTOR ──────────────────────────────────────────────────────────
LIVE_MODULES = [
    "📈  Sales Intelligence",
    "📦  Inventory Intelligence",
    "👥  HR & Payroll",
    "💰  Finance & Accounting",
    "🛒  Retail & E-commerce",
    "🚚  Logistics",
    "🍽️  Restaurant",
    "🏥  Healthcare",
    "🏭  Manufacturing",
    "📣  Marketing",
    "🎓  Education",
    "🏨  Hospitality",
    "🌾  Agriculture",
    "🏗️  Construction",
    "🏦  Banking",
]
SOON_MODULES = ["⚖️  Legal", "🌍  NGO", "📡  Telecom", "🏘️  Real Estate"]

if "module" not in st.session_state:
    st.session_state.module = "Sales Intelligence"

# Map display label → internal name
def clean_module(label):
    return label.split("  ", 1)[1] if "  " in label else label

module_choice = st.radio(
    "module_selector",
    LIVE_MODULES,
    index=[clean_module(m) for m in LIVE_MODULES].index(st.session_state.module) if st.session_state.module in [clean_module(m) for m in LIVE_MODULES] else 0,
    horizontal=True,
    label_visibility="collapsed"
)
module = clean_module(module_choice)
st.session_state.module = module

# Show coming soon pills
st.markdown(
    "<div style='display:flex;gap:8px;flex-wrap:wrap;margin:-8px 0 12px;'>" +
    "".join([f"<span style='display:inline-flex;align-items:center;gap:5px;padding:7px 14px;border-radius:100px;border:1.5px dashed #d1d5db;background:#f9fafb;font-size:0.78rem;font-weight:600;color:#9ca3af;'>{m} <span style='background:#e5e7eb;color:#6b7280;font-size:0.62rem;font-weight:700;padding:1px 6px;border-radius:100px;'>SOON</span></span>" for m in SOON_MODULES]) +
    "</div>",
    unsafe_allow_html=True
)

st.markdown("""
<div style='margin:12px 0 16px;'>
    <span class='fpill'>⚡ Auto-cleaning</span>
    <span class='fpill'>📈 Forecasting</span>
    <span class='fpill'>🚨 Smart alerts</span>
    <span class='fpill'>📄 PDF report</span>
    <span class='fpill'>🔒 Never stored</span>
    <span class='fpill'>📊 1M+ rows supported</span>
    <span class='fpill'>🗂️ Excel · CSV · JSON · Google Sheets</span>
</div>
""", unsafe_allow_html=True)

# ── DATA SOURCE SELECTOR ─────────────────────────────────────────────────────
src_tab1, src_tab2 = st.tabs(["📁 Upload File", "🔗 Google Sheets URL"])

with src_tab1:
    uploaded_file = st.file_uploader(
        "Upload your data file",
        type=["xlsx","csv","json"],
        label_visibility="collapsed",
        help="Supports Excel (.xlsx), CSV (.csv), JSON (.json) · Up to 200MB · Any number of rows"
    )
    gsheet_df = None

with src_tab2:
    st.markdown("""
    <div style='background:#eef2ff;border-radius:10px;padding:14px 16px;margin-bottom:12px;font-size:0.85rem;color:#3730a3;border-left:3px solid #6366f1;'>
    <b>How to share your Google Sheet:</b><br/>
    Open Google Sheets → <b>File → Share → Anyone with the link → Viewer</b><br/>
    Then paste the link below ↓
    </div>
    """, unsafe_allow_html=True)

    gsheet_url = st.text_input(
        "Paste Google Sheets URL",
        placeholder="https://docs.google.com/spreadsheets/d/...",
        label_visibility="collapsed"
    )

    gsheet_df = None
    if gsheet_url and gsheet_url.strip():
        try:
            # Convert share URL to CSV export URL
            def gsheet_to_csv_url(url):
                import re
                # Extract sheet ID
                match = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
                if not match:
                    return None, None
                sheet_id = match.group(1)
                # Extract gid (sheet tab) if present
                gid_match = re.search(r'gid=(\d+)', url)
                gid = gid_match.group(1) if gid_match else '0'
                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
                return csv_url, sheet_id

            csv_url, sheet_id = gsheet_to_csv_url(gsheet_url.strip())
            if not csv_url:
                st.error("❌ Invalid Google Sheets URL. Make sure you copy the full link.")
            else:
                with st.spinner("⚡ Connecting to Google Sheets..."):
                    response_df = pd.read_csv(csv_url)
                    if len(response_df) > 0:
                        gsheet_df = response_df
                        st.success(f"✅ Connected! {len(gsheet_df):,} rows · {len(gsheet_df.columns)} columns loaded from Google Sheets")
                    else:
                        st.warning("⚠️ Sheet appears empty. Check the URL and sharing settings.")
        except Exception as e:
            err = str(e)
            if '403' in err or 'Permission' in err or 'Forbidden' in err:
                st.error("❌ Access denied. Make sure the sheet is shared as **Anyone with the link → Viewer**.")
            elif '404' in err or 'Not Found' in err:
                st.error("❌ Sheet not found. Check the URL is correct.")
            else:
                st.error(f"❌ Could not load sheet. Make sure it's publicly shared. ({err[:80]})")

# ── Resolve active data source ────────────────────────────────────────────────
# Priority: Google Sheets > uploaded file
if gsheet_df is not None:
    _active_file = gsheet_df
    _active_name = "Google Sheets"
elif uploaded_file is not None:
    _active_file = uploaded_file
    _active_name = uploaded_file.name
else:
    _active_file = None
    _active_name = None

# ── HELPER: section header ───────────────────────────────────────────────────
def sh(icon, title, badge=None):
    badge_html = f"<span class='sh-badge'>{badge}</span>" if badge else ""
    st.markdown(f"<div class='sh'><span class='sh-icon'>{icon}</span><span class='sh-title'>{title}</span>{badge_html}</div>", unsafe_allow_html=True)

def abox(msg, kind="amber"):
    st.markdown(f"<div class='abox {kind}'>{msg}</div>", unsafe_allow_html=True)

def kpi_row(items):
    # items = list of (value, label, color, sub)
    cols = st.columns(len(items))
    for col, (val, label, color, sub) in zip(cols, items):
        with col:
            sub_html = f"<div class='kpi-sub' style='color:{'#ef4444' if 'red' in color else '#10b981' if 'green' in color else '#f59e0b' if 'amber' in color else '#6366f1'}'>{sub}</div>" if sub else ""
            st.markdown(f"""
            <div class='kpi-card {color}'>
                <div class='kpi-num'>{val}</div>
                <div class='kpi-label'>{label}</div>
                {sub_html}
            </div>""", unsafe_allow_html=True)

def clean_chart(fig, ax):
    ax.set_facecolor('#f8fafc')
    fig.patch.set_facecolor('white')
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#e5e7eb')
    ax.tick_params(colors='#6b7280', labelsize=9)
    ax.yaxis.label.set_color('#6b7280')
    ax.xaxis.label.set_color('#6b7280')

# ════════════════════════════════════════════════════════════════════════════
# SHOW LANDING IF NO FILE
# ════════════════════════════════════════════════════════════════════════════
if _active_file is None:
    st.markdown("""
    <div class='upload-hint'>
        <div style='font-size:2.5rem'>⚡</div>
        <h3>Upload your data or connect Google Sheets</h3>
        <p>Excel · CSV · JSON · Google Sheets · Any size · Results in 60 seconds</p>
        <div style='display:flex; justify-content:center; gap:40px; flex-wrap:wrap; margin-top:24px;'>
            <div style='max-width:180px; text-align:left;'>
                <div style='font-weight:700; color:#0a0a0a; margin-bottom:6px;'>📊 Instant Analysis</div>
                <div style='color:#6b7280; font-size:0.83rem; line-height:1.7;'>Revenue · Products · Trends · Forecasts</div>
            </div>
            <div style='max-width:180px; text-align:left;'>
                <div style='font-weight:700; color:#0a0a0a; margin-bottom:6px;'>👤 People Insights</div>
                <div style='color:#6b7280; font-size:0.83rem; line-height:1.7;'>Salesperson · HR · Payroll · Performance</div>
            </div>
            <div style='max-width:180px; text-align:left;'>
                <div style='font-weight:700; color:#0a0a0a; margin-bottom:6px;'>🔮 Predictions</div>
                <div style='color:#6b7280; font-size:0.83rem; line-height:1.7;'>6-month forecast · Smart alerts · PDF</div>
            </div>
        </div>
    </div>
    <p style='text-align:center; color:#9ca3af; font-size:0.8rem; margin-top:14px;'>🔒 Your data is processed locally and deleted immediately. Never stored. Never shared.</p>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load & Smart Clean ───────────────────────────────────────────────────────
if gsheet_df is not None:
    # Google Sheets already loaded as DataFrame
    df_raw = gsheet_df.copy()
    df, fixes = smart_clean(df_raw, "Google Sheets")
else:
    df_raw = load_file(uploaded_file)
    df, fixes = smart_clean(df_raw, uploaded_file.name)

# ════════════════════════════════════════════════════════════════════════════
# MODULE: INVENTORY INTELLIGENCE
# ════════════════════════════════════════════════════════════════════════════
if module == "Inventory Intelligence":
    df_inv = df.copy()

    prod_col      = next((c for c in df_inv.columns if 'product name' in c.lower() or 'product' in c.lower()), None)
    cat_col       = next((c for c in df_inv.columns if 'category' in c.lower()), None)
    stock_col     = next((c for c in df_inv.columns if 'current stock' in c.lower() or 'stock' in c.lower()), None)
    reorder_col   = next((c for c in df_inv.columns if 'reorder' in c.lower()), None)
    max_col       = next((c for c in df_inv.columns if 'max stock' in c.lower()), None)
    cost_col      = next((c for c in df_inv.columns if 'unit cost' in c.lower() or 'cost' in c.lower()), None)
    price_col     = next((c for c in df_inv.columns if 'selling price' in c.lower() or 'price' in c.lower()), None)
    demand_col    = next((c for c in df_inv.columns if 'monthly demand' in c.lower() or 'demand' in c.lower()), None)
    sold30_col    = next((c for c in df_inv.columns if 'last 30' in c.lower()), None)
    sold90_col    = next((c for c in df_inv.columns if 'last 90' in c.lower()), None)
    last_sold_col = next((c for c in df_inv.columns if 'last sold' in c.lower()), None)
    ware_col      = next((c for c in df_inv.columns if 'warehouse' in c.lower()), None)
    supplier_col  = next((c for c in df_inv.columns if 'supplier' in c.lower()), None)

    if stock_col and reorder_col:
        df_inv['Stock Status'] = 'Normal'
        df_inv.loc[df_inv[stock_col] == 0, 'Stock Status'] = 'Out of Stock'
        df_inv.loc[(df_inv[stock_col] > 0) & (df_inv[stock_col] <= df_inv[reorder_col]), 'Stock Status'] = 'Low Stock'
        if max_col and max_col in df_inv.columns:
            df_inv.loc[df_inv[stock_col] >= df_inv[max_col] * 0.9, 'Stock Status'] = 'Overstocked'
    if stock_col and demand_col:
        df_inv['Days of Stock Left'] = (df_inv[stock_col] / (df_inv[demand_col] / 30)).round(1)
        df_inv['Days of Stock Left'] = df_inv['Days of Stock Left'].replace([float('inf'), -float('inf')], 999)
    if cost_col and price_col:
        df_inv['Margin ($)'] = df_inv[price_col] - df_inv[cost_col]
        df_inv['Margin %'] = ((df_inv['Margin ($)'] / df_inv[price_col]) * 100).round(1)
    if stock_col and cost_col:
        df_inv['Stock Value ($)'] = (df_inv[stock_col] * df_inv[cost_col]).round(0)
    dead_stock = pd.DataFrame()
    if last_sold_col and stock_col:
        dead_stock = df_inv[(df_inv[last_sold_col] > 90) & (df_inv[stock_col] > 0)].copy()

    show_clean_report(fixes, df_inv, df_raw)

    with st.sidebar:
        st.markdown("### 🔍 Filter Inventory")
        df_filtered = df_inv.copy()
        if cat_col:
            cats = ['All'] + sorted(df_inv[cat_col].dropna().unique().tolist())
            sc = st.selectbox("Category", cats, key="inv_cat")
            if sc != 'All': df_filtered = df_filtered[df_filtered[cat_col] == sc]
        if ware_col:
            wares = ['All'] + sorted(df_inv[ware_col].dropna().unique().tolist())
            sw = st.selectbox("Warehouse", wares, key="inv_ware")
            if sw != 'All': df_filtered = df_filtered[df_filtered[ware_col] == sw]
        if 'Stock Status' in df_inv.columns:
            statuses = ['All'] + sorted(df_inv['Stock Status'].dropna().unique().tolist())
            ss = st.selectbox("Status", statuses, key="inv_status")
            if ss != 'All': df_filtered = df_filtered[df_filtered['Stock Status'] == ss]
        st.caption(f"**{len(df_filtered)}** of **{len(df_inv)}** products")

    inv_val    = df_filtered['Stock Value ($)'].sum() if 'Stock Value ($)' in df_filtered.columns else 0
    oos_count  = (df_filtered[stock_col] == 0).sum() if stock_col else 0
    low_count  = (df_filtered['Stock Status'] == 'Low Stock').sum() if 'Stock Status' in df_filtered.columns else 0
    over_count = (df_filtered['Stock Status'] == 'Overstocked').sum() if 'Stock Status' in df_filtered.columns else 0
    dead_count = len(df_filtered[(df_filtered[last_sold_col] > 90) & (df_filtered[stock_col] > 0)]) if last_sold_col and stock_col else 0

    total_skus = df_filtered[prod_col].nunique() if prod_col else len(df_filtered)
    kpi_row([
        (f"{total_skus}", "Total SKUs", "indigo", None),
        (f"${inv_val:,.0f}", "Inventory Value", "indigo", None),
        (f"{oos_count}", "Out of Stock", "red", "⚠️ Critical" if oos_count > 0 else "✅ Clear"),
        (f"{low_count}", "Low Stock", "amber", "⚠️ Reorder soon" if low_count > 0 else "✅ OK"),
        (f"{over_count}", "Overstocked", "purple", "Capital tied up" if over_count > 0 else "✅ OK"),
        (f"{dead_count}", "Dead Stock", "amber", "Review needed" if dead_count > 0 else "✅ None"),
    ])

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🚦 Stock Status","💀 Dead Stock","🏆 ABC Analysis","🔔 Reorder Alerts","📊 Category & Margins","🚨 Smart Alerts"])

    with tab1:
        if 'Stock Status' in df_filtered.columns:
            col_a, col_b = st.columns(2)
            with col_a:
                status_counts = df_filtered['Stock Status'].value_counts()
                colors_map = {'Out of Stock':'#ef4444','Low Stock':'#f59e0b','Normal':'#10b981','Overstocked':'#6366f1'}
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                bars = ax.bar(status_counts.index, status_counts.values,
                              color=[colors_map.get(s,'#6366f1') for s in status_counts.index],
                              width=0.5, edgecolor='white', linewidth=2)
                ax.set_title("Stock Status Distribution", fontweight='bold', fontsize=12)
                for bar, val in zip(bars, status_counts.values):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2, str(val), ha='center', va='bottom', fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                for s, c in [('Out of Stock','#ef4444'),('Low Stock','#f59e0b'),('Normal','#10b981'),('Overstocked','#6366f1')]:
                    cnt = (df_filtered['Stock Status']==s).sum()
                    pct = cnt/len(df_filtered)*100 if len(df_filtered) > 0 else 0
                    st.markdown(f"<div style='display:flex;justify-content:space-between;background:white;border-radius:10px;padding:12px 16px;margin:6px 0;border-left:4px solid {c};box-shadow:0 1px 4px rgba(0,0,0,0.05);'><span style='font-weight:600'>{s}</span><span style='font-weight:800;color:{c}'>{cnt} <span style='font-size:0.75rem;color:#6b7280'>({pct:.0f}%)</span></span></div>", unsafe_allow_html=True)
            urgent = df_filtered[df_filtered['Stock Status'].isin(['Out of Stock','Low Stock'])]
            if len(urgent) > 0:
                sh("📋","Products Needing Action", f"{len(urgent)} items")
                show_cols = [c for c in [prod_col,cat_col,ware_col,stock_col,reorder_col,'Stock Status'] if c and c in urgent.columns]
                st.dataframe(urgent[show_cols].sort_values('Stock Status').reset_index(drop=True), use_container_width=True, height=350)

    with tab2:
        dead_f = df_filtered[(df_filtered[last_sold_col]>90)&(df_filtered[stock_col]>0)].copy() if last_sold_col and stock_col else pd.DataFrame()
        dead_val = dead_f['Stock Value ($)'].sum() if 'Stock Value ($)' in dead_f.columns else 0
        if len(dead_f) > 0:
            kpi_row([(f"{len(dead_f)}","Dead Products","amber",None),(f"${dead_val:,.0f}","Capital Locked","red","Liquidate or discount"),(f"{dead_val/inv_val*100:.1f}%","% of Inventory","amber",None)])
            abox(f"⚠️ <b>{len(dead_f)} products</b> haven't sold in 90+ days. <b>${dead_val:,.0f}</b> locked in dead stock. Consider discounting or bundling.", "amber")
            show_cols = [c for c in [prod_col,cat_col,ware_col,stock_col,last_sold_col,'Stock Value ($)'] if c and c in dead_f.columns]
            st.dataframe(dead_f[show_cols].sort_values('Stock Value ($)', ascending=False).reset_index(drop=True), use_container_width=True, height=350)
        else:
            abox("✅ No dead stock detected — all inventory is moving well!", "green")

    with tab3:
        if prod_col and sold90_col:
            abc = df_filtered.groupby(prod_col)[sold90_col].sum().sort_values(ascending=False).reset_index()
            abc.columns = ['Product','Units Sold (90d)']
            total_s = abc['Units Sold (90d)'].sum()
            abc['Cumulative %'] = (abc['Units Sold (90d)'].cumsum()/total_s*100).round(1) if total_s > 0 else 0
            abc['ABC Class'] = 'C'
            abc.loc[abc['Cumulative %'] <= 80, 'ABC Class'] = 'A'
            abc.loc[(abc['Cumulative %'] > 80) & (abc['Cumulative %'] <= 95), 'ABC Class'] = 'B'
            st.caption("**A** = Top products driving 80% of sales · **B** = Middle tier · **C** = Slow movers")
            col_a, col_b = st.columns(2)
            with col_a:
                abc_c = abc['ABC Class'].value_counts().reindex(['A','B','C'], fill_value=0)
                fig, ax = plt.subplots(figsize=(5,5))
                clean_chart(fig, ax)
                ax.pie(abc_c.values, labels=[f'Class {c}' for c in abc_c.index],
                       autopct='%1.0f%%', colors=['#10b981','#6366f1','#9ca3af'],
                       wedgeprops=dict(edgecolor='white', linewidth=2), startangle=90)
                ax.set_title("ABC Distribution", fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                for cls, col, lbl in [('A','#10b981','Star Products — Never let OOS'),('B','#6366f1','Growing — Monitor closely'),('C','#9ca3af','Slow — Review or discount')]:
                    cd = abc[abc['ABC Class']==cls]
                    st.markdown(f"<div style='background:white;border-radius:10px;padding:13px 16px;margin:8px 0;border-left:4px solid {col};box-shadow:0 1px 4px rgba(0,0,0,0.05);'><div style='font-weight:800;color:{col}'>Class {cls} — {len(cd)} products</div><div style='font-size:0.8rem;color:#6b7280;margin-top:2px;'>{lbl}</div><div style='font-size:0.82rem;color:#0a0a0a;margin-top:4px;font-weight:600;'>{cd['Units Sold (90d)'].sum():,} units sold</div></div>", unsafe_allow_html=True)
            st.dataframe(abc.style.apply(lambda col: ['background-color:#f0fdf4;color:#166534' if v=='A' else 'background-color:#eef2ff;color:#3730a3' if v=='B' else '' for v in col], subset=['ABC Class']), use_container_width=True, height=350)

    with tab4:
        if 'Days of Stock Left' in df_filtered.columns:
            reorder_df = df_filtered[df_filtered['Days of Stock Left'] < 30].sort_values('Days of Stock Left')
            crit_df = df_filtered[df_filtered['Days of Stock Left'] < 7]
            kpi_row([(f"{len(reorder_df)}","Reorder Needed","amber","< 30 days left"),(f"{len(crit_df)}","Critical Urgent","red","< 7 days left"),(f"{len(df_filtered)-len(reorder_df)}","Healthy","green","> 30 days")])
            if len(reorder_df) > 0:
                abox(f"⚠️ <b>{len(reorder_df)} products</b> will run out within 30 days. Place reorders now!", "amber")
                top15 = reorder_df.head(15)
                if prod_col in top15.columns:
                    fig, ax = plt.subplots(figsize=(10,4))
                    clean_chart(fig, ax)
                    bar_c = ['#ef4444' if d < 7 else '#f59e0b' for d in top15['Days of Stock Left']]
                    ax.barh(top15[prod_col].str[:28], top15['Days of Stock Left'], color=bar_c)
                    ax.axvline(x=7, color='#ef4444', linestyle='--', alpha=0.5, label='Critical (7d)')
                    ax.axvline(x=14, color='#f59e0b', linestyle='--', alpha=0.5, label='Warning (14d)')
                    ax.set_title("Days of Stock Remaining", fontweight='bold')
                    ax.legend(fontsize=9)
                    plt.tight_layout(); st.pyplot(fig); plt.close()
                show_cols = [c for c in [prod_col,cat_col,ware_col,stock_col,demand_col,'Days of Stock Left',supplier_col] if c and c in reorder_df.columns]
                st.dataframe(reorder_df[show_cols].reset_index(drop=True), use_container_width=True, height=300)

    with tab5:
        col_a, col_b = st.columns(2)
        with col_a:
            if cat_col and 'Stock Value ($)' in df_filtered.columns:
                sh("📦","Stock Value by Category")
                cat_val = df_filtered.groupby(cat_col)['Stock Value ($)'].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                colors_g = ['#6366f1','#818cf8','#a5b4fc','#c7d2fe','#e0e7ff','#4f46e5']
                ax.barh(cat_val.index, cat_val.values, color=colors_g[:len(cat_val)])
                ax.invert_yaxis()
                for i, val in enumerate(cat_val.values):
                    ax.text(val, i, f'  ${val:,.0f}', va='center', fontsize=9)
                plt.tight_layout(); st.pyplot(fig); plt.close()
        with col_b:
            if cat_col and sold90_col:
                sh("📊","Units Sold by Category (90d)")
                cat_sold = df_filtered.groupby(cat_col)[sold90_col].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                ax.pie(cat_sold.values, labels=cat_sold.index, autopct='%1.1f%%',
                       colors=['#6366f1','#818cf8','#a5b4fc','#c7d2fe','#e0e7ff','#4f46e5'],
                       wedgeprops=dict(edgecolor='white', linewidth=2), startangle=90)
                plt.tight_layout(); st.pyplot(fig); plt.close()
        if 'Margin %' in df_filtered.columns and prod_col:
            col_c, col_d = st.columns(2)
            with col_c:
                sh("🏆","Top 10 Highest Margin")
                tm = df_filtered[[prod_col,cost_col,price_col,'Margin %']].sort_values('Margin %', ascending=False).head(10).reset_index(drop=True)
                st.dataframe(tm.style.background_gradient(subset=['Margin %'], cmap='Greens'), use_container_width=True)
            with col_d:
                sh("⚠️","Bottom 10 Lowest Margin")
                lm = df_filtered[[prod_col,cost_col,price_col,'Margin %']].sort_values('Margin %').head(10).reset_index(drop=True)
                st.dataframe(lm.style.background_gradient(subset=['Margin %'], cmap='Reds_r'), use_container_width=True)

    with tab6:
        alerts = []
        if stock_col and (df_filtered[stock_col]==0).sum() > 0:
            alerts.append(('red', f"🔴 <b>{(df_filtered[stock_col]==0).sum()} products</b> are OUT OF STOCK — immediate action required"))
        if 'Days of Stock Left' in df_filtered.columns and (df_filtered['Days of Stock Left']<7).sum() > 0:
            alerts.append(('red', f"🔴 <b>{(df_filtered['Days of Stock Left']<7).sum()} products</b> will run out in 7 days — URGENT reorder"))
        if dead_count > 0:
            dead_f2 = df_filtered[(df_filtered[last_sold_col]>90)&(df_filtered[stock_col]>0)] if last_sold_col and stock_col else pd.DataFrame()
            dv2 = dead_f2['Stock Value ($)'].sum() if 'Stock Value ($)' in dead_f2.columns else 0
            alerts.append(('amber', f"⚠️ <b>{dead_count} dead stock products</b> — ${dv2:,.0f} locked. Discount or liquidate."))
        if 'Stock Status' in df_filtered.columns and (df_filtered['Stock Status']=='Overstocked').sum() > 0:
            alerts.append(('amber', f"⚠️ <b>{(df_filtered['Stock Status']=='Overstocked').sum()} products</b> overstocked — capital tied up unnecessarily"))
        if 'Margin %' in df_filtered.columns and (df_filtered['Margin %']<0).sum() > 0:
            alerts.append(('red', f"🔴 <b>{(df_filtered['Margin %']<0).sum()} products</b> have NEGATIVE margins — selling below cost!"))
        if not alerts:
            abox("✅ All clear! Inventory looks healthy — no critical alerts.", "green")
        else:
            for kind, msg in alerts:
                abox(msg, kind)

    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# MODULE: HR & PAYROLL INTELLIGENCE
# ════════════════════════════════════════════════════════════════════════════
if module == "HR & Payroll":
    df_hr = df.copy()

    # Detect columns
    name_col    = next((c for c in df_hr.columns if 'name' in c.lower() and 'employee' in c.lower()), None) or \
                  next((c for c in df_hr.columns if 'name' in c.lower()), None)
    dept_col    = next((c for c in df_hr.columns if 'dept' in c.lower() or 'department' in c.lower()), None)
    desig_col   = next((c for c in df_hr.columns if 'desig' in c.lower() or 'title' in c.lower() or 'role' in c.lower()), None)
    salary_col  = next((c for c in df_hr.columns if 'salary' in c.lower() or 'ctc' in c.lower() or 'pay' in c.lower()), None)
    gender_col  = next((c for c in df_hr.columns if 'gender' in c.lower() or 'sex' in c.lower()), None)
    age_col     = next((c for c in df_hr.columns if 'age' in c.lower()), None)
    exp_col     = next((c for c in df_hr.columns if 'experience' in c.lower() or 'exp' in c.lower() or 'tenure' in c.lower()), None)
    perf_col    = next((c for c in df_hr.columns if 'performance' in c.lower() or 'rating' in c.lower() or 'score' in c.lower()), None)
    status_col  = next((c for c in df_hr.columns if 'status' in c.lower() or 'active' in c.lower()), None)
    leave_col   = next((c for c in df_hr.columns if 'leave' in c.lower()), None)
    overtime_col= next((c for c in df_hr.columns if 'overtime' in c.lower() or 'ot' in c.lower()), None)
    join_col    = next((c for c in df_hr.columns if 'join' in c.lower() or 'start' in c.lower() or 'hire' in c.lower()), None)
    loc_col     = next((c for c in df_hr.columns if 'location' in c.lower() or 'city' in c.lower() or 'office' in c.lower()), None)

    show_clean_report(fixes, df_hr, df_raw)

    # Sidebar filters
    with st.sidebar:
        st.markdown("### 🔍 Filter Employees")
        df_hf = df_hr.copy()
        if dept_col:
            depts = ['All Departments'] + sorted(df_hr[dept_col].dropna().unique().tolist())
            sd = st.selectbox("Department", depts, key="hr_dept")
            if sd != 'All Departments': df_hf = df_hf[df_hf[dept_col] == sd]
        if gender_col:
            genders = ['All'] + sorted(df_hr[gender_col].dropna().unique().tolist())
            sg = st.selectbox("Gender", genders, key="hr_gender")
            if sg != 'All': df_hf = df_hf[df_hf[gender_col] == sg]
        if status_col:
            statuses = ['All'] + sorted(df_hr[status_col].dropna().unique().tolist())
            ss = st.selectbox("Status", statuses, key="hr_status")
            if ss != 'All': df_hf = df_hf[df_hf[status_col] == ss]
        if loc_col:
            locs = ['All'] + sorted(df_hr[loc_col].dropna().unique().tolist())
            sl = st.selectbox("Location", locs, key="hr_loc")
            if sl != 'All': df_hf = df_hf[df_hf[loc_col] == sl]
        st.caption(f"**{len(df_hf)}** of **{len(df_hr)}** employees")

    # KPIs
    total_emp  = len(df_hf)
    total_pay  = nc(df_hf[salary_col]).sum() if salary_col else 0
    avg_salary = nc(df_hf[salary_col]).mean() or 0 if salary_col else 0
    num_depts  = df_hf[dept_col].nunique() if dept_col else 0
    avg_perf   = nc(df_hf[perf_col]).mean() or 0 if perf_col else 0
    ot_alert   = (df_hf[overtime_col] > 20).sum() if overtime_col else 0

    kpi_row([
        (f"{total_emp}", "Total Employees", "indigo", None),
        (f"${total_pay:,.0f}", "Total Payroll", "indigo", "Per month"),
        (f"${avg_salary:,.0f}", "Avg Salary", "indigo", None),
        (f"{num_depts}", "Departments", "indigo", None),
        (f"{avg_perf:.1f}" if avg_perf else "N/A", "Avg Performance", "green", "Out of 10" if avg_perf else None),
        (f"{ot_alert}", "Overtime Risk", "amber" if ot_alert > 0 else "green", "Review workload" if ot_alert > 0 else "All good"),
    ])

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "👥 Headcount","💵 Payroll Analysis","⭐ Performance","⚠️ Attrition Risk","📊 Demographics","🚨 Smart Alerts"
    ])

    with tab1:
        sh("👥", "Headcount by Department")
        if dept_col:
            col_a, col_b = st.columns(2)
            with col_a:
                dept_count = df_hf[dept_col].value_counts()
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                bars = ax.barh(dept_count.index, dept_count.values, color='#6366f1')
                ax.invert_yaxis()
                for i, val in enumerate(dept_count.values):
                    ax.text(val+0.1, i, str(val), va='center', fontweight='bold', fontsize=10)
                ax.set_title("Employees per Department", fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                if gender_col:
                    g_count = df_hf[gender_col].value_counts()
                    fig, ax = plt.subplots(figsize=(5,5))
                    clean_chart(fig, ax)
                    ax.pie(g_count.values, labels=g_count.index, autopct='%1.1f%%',
                           colors=['#6366f1','#f472b6','#34d399'],
                           wedgeprops=dict(edgecolor='white', linewidth=2))
                    ax.set_title("Gender Distribution", fontweight='bold')
                    plt.tight_layout(); st.pyplot(fig); plt.close()
                elif loc_col:
                    loc_count = df_hf[loc_col].value_counts()
                    fig, ax = plt.subplots(figsize=(5,5))
                    clean_chart(fig, ax)
                    ax.pie(loc_count.values, labels=loc_count.index, autopct='%1.1f%%',
                           colors=['#6366f1','#818cf8','#a5b4fc','#c7d2fe','#4f46e5'],
                           wedgeprops=dict(edgecolor='white', linewidth=2))
                    ax.set_title("Employees by Location", fontweight='bold')
                    plt.tight_layout(); st.pyplot(fig); plt.close()

        if desig_col:
            sh("🏷️","Headcount by Designation")
            desig_count = df_hf[desig_col].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(10,3))
            clean_chart(fig, ax)
            ax.bar(desig_count.index, desig_count.values, color='#818cf8', width=0.6)
            plt.xticks(rotation=30, ha='right', fontsize=9)
            for i, (idx, val) in enumerate(desig_count.items()):
                ax.text(i, val+0.1, str(val), ha='center', fontweight='bold', fontsize=10)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        if salary_col:
            sh("💵","Payroll Analysis")
            col_a, col_b = st.columns(2)
            with col_a:
                if dept_col:
                    dept_pay = df_hf.groupby(dept_col)[salary_col].sum().sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(6,4))
                    clean_chart(fig, ax)
                    ax.barh(dept_pay.index, dept_pay.values, color='#6366f1')
                    ax.invert_yaxis()
                    ax.set_title("Payroll Cost by Department", fontweight='bold')
                    for i, val in enumerate(dept_pay.values):
                        ax.text(val, i, f'  ${val:,.0f}', va='center', fontsize=9)
                    plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                ax.hist(df_hf[salary_col].dropna(), bins=15, color='#818cf8', edgecolor='white', linewidth=1.5)
                ax.axvline(avg_salary, color='#ef4444', linestyle='--', linewidth=2, label=f'Avg: ${avg_salary:,.0f}')
                ax.set_title("Salary Distribution", fontweight='bold')
                ax.set_xlabel("Salary ($)")
                ax.legend(fontsize=9)
                plt.tight_layout(); st.pyplot(fig); plt.close()

            # Salary outliers
            sh("🔍","Salary Outliers")
            q1 = df_hf[salary_col].quantile(0.25)
            q3 = df_hf[salary_col].quantile(0.75)
            iqr = q3 - q1
            outliers = df_hf[(df_hf[salary_col] < q1 - 1.5*iqr) | (df_hf[salary_col] > q3 + 1.5*iqr)]
            if len(outliers) > 0:
                abox(f"⚠️ <b>{len(outliers)} salary outliers</b> detected — salaries significantly above or below team average.", "amber")
                show_cols = [c for c in [name_col, dept_col, desig_col, salary_col] if c and c in outliers.columns]
                st.dataframe(outliers[show_cols].reset_index(drop=True), use_container_width=True)
            else:
                abox("✅ Salary distribution is healthy — no major outliers.", "green")

            # Dept comparison table
            if dept_col:
                sh("📋","Department Salary Summary")
                dept_summary = df_hf.groupby(dept_col)[salary_col].agg(['count','mean','min','max','sum']).reset_index()
                dept_summary.columns = ['Department','Headcount','Avg Salary','Min Salary','Max Salary','Total Payroll']
                dept_summary = dept_summary.round(0).sort_values('Total Payroll', ascending=False)
                st.dataframe(dept_summary.style.background_gradient(subset=['Total Payroll'], cmap='Blues'), use_container_width=True)

    with tab3:
        if perf_col:
            sh("⭐","Performance Analysis")
            col_a, col_b = st.columns(2)
            with col_a:
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                ax.hist(df_hf[perf_col].dropna(), bins=10, color='#10b981', edgecolor='white', linewidth=1.5)
                ax.axvline(avg_perf, color='#ef4444', linestyle='--', linewidth=2, label=f'Avg: {avg_perf:.1f}')
                ax.set_title("Performance Score Distribution", fontweight='bold')
                ax.legend(fontsize=9)
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                if dept_col:
                    dept_perf = df_hf.groupby(dept_col)[perf_col].mean().sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(6,4))
                    clean_chart(fig, ax)
                    bar_colors = ['#10b981' if v >= avg_perf else '#ef4444' for v in dept_perf.values]
                    ax.barh(dept_perf.index, dept_perf.values, color=bar_colors)
                    ax.axvline(avg_perf, color='#6366f1', linestyle='--', linewidth=1.5, alpha=0.7, label='Company Avg')
                    ax.invert_yaxis()
                    ax.set_title("Avg Performance by Department", fontweight='bold')
                    ax.legend(fontsize=9)
                    plt.tight_layout(); st.pyplot(fig); plt.close()

            col_c, col_d = st.columns(2)
            with col_c:
                sh("🏆","Top 10 Performers")
                show_cols = [c for c in [name_col, dept_col, desig_col, salary_col, perf_col] if c and c in df_hf.columns]
                top_p = df_hf[show_cols].sort_values(perf_col, ascending=False).head(10).reset_index(drop=True)
                st.dataframe(top_p, use_container_width=True)
            with col_d:
                sh("⚠️","Bottom 10 — Need Support")
                bot_p = df_hf[show_cols].sort_values(perf_col).head(10).reset_index(drop=True)
                st.dataframe(bot_p, use_container_width=True)
        else:
            abox("ℹ️ No performance score column detected. Add a 'Performance Score' or 'Rating' column for this analysis.", "blue")

    with tab4:
        sh("⚠️","Attrition Risk Detection")
        df_hf['Risk Score'] = 0
        risk_reasons = {}

        if overtime_col:
            ot_mask = df_hf[overtime_col] > 20
            df_hf.loc[ot_mask, 'Risk Score'] += 2
            for idx in df_hf[ot_mask].index:
                risk_reasons[idx] = risk_reasons.get(idx, []) + ["High overtime"]

        if perf_col:
            low_perf_mask = df_hf[perf_col] < df_hf[perf_col].quantile(0.25)
            df_hf.loc[low_perf_mask, 'Risk Score'] += 2
            for idx in df_hf[low_perf_mask].index:
                risk_reasons[idx] = risk_reasons.get(idx, []) + ["Low performance"]

        if leave_col:
            high_leave_mask = df_hf[leave_col] > df_hf[leave_col].quantile(0.75)
            df_hf.loc[high_leave_mask, 'Risk Score'] += 1
            for idx in df_hf[high_leave_mask].index:
                risk_reasons[idx] = risk_reasons.get(idx, []) + ["High leave usage"]

        if salary_col and dept_col:
            dept_avg = df_hf.groupby(dept_col)[salary_col].transform('mean')
            underpaid_mask = df_hf[salary_col] < dept_avg * 0.8
            df_hf.loc[underpaid_mask, 'Risk Score'] += 2
            for idx in df_hf[underpaid_mask].index:
                risk_reasons[idx] = risk_reasons.get(idx, []) + ["Underpaid vs dept avg"]

        high_risk = df_hf[df_hf['Risk Score'] >= 3].copy()
        med_risk   = df_hf[(df_hf['Risk Score'] >= 1) & (df_hf['Risk Score'] < 3)].copy()

        kpi_row([
            (f"{len(high_risk)}", "High Risk", "red", "Likely to leave"),
            (f"{len(med_risk)}", "Medium Risk", "amber", "Watch closely"),
            (f"{len(df_hf)-len(high_risk)-len(med_risk)}", "Low Risk", "green", "Stable"),
        ])

        if len(high_risk) > 0:
            abox(f"🔴 <b>{len(high_risk)} employees</b> are at HIGH attrition risk. Take action immediately — engagement, salary review, or 1:1 meeting.", "red")
            show_cols = [c for c in [name_col, dept_col, desig_col, salary_col, perf_col, overtime_col, 'Risk Score'] if c and c in high_risk.columns]
            high_risk['Risk Reasons'] = high_risk.index.map(lambda x: ', '.join(risk_reasons.get(x, [])))
            st.dataframe(high_risk[show_cols + ['Risk Reasons']].sort_values('Risk Score', ascending=False).reset_index(drop=True), use_container_width=True)
        else:
            abox("✅ No high attrition risk employees detected.", "green")

    with tab5:
        sh("📊","Demographics Analysis")
        col_a, col_b = st.columns(2)
        with col_a:
            if age_col:
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                ax.hist(df_hf[age_col].dropna(), bins=10, color='#6366f1', edgecolor='white', linewidth=1.5)
                ax.axvline(nc(df_hf[age_col]).mean() or 0, color='#ef4444', linestyle='--', label=f"Avg: {nc(df_hf[age_col]).mean() or 0:.0f} yrs")
                ax.set_title("Age Distribution", fontweight='bold')
                ax.set_xlabel("Age")
                ax.legend()
                plt.tight_layout(); st.pyplot(fig); plt.close()
        with col_b:
            if exp_col:
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                ax.hist(df_hf[exp_col].dropna(), bins=10, color='#818cf8', edgecolor='white', linewidth=1.5)
                ax.axvline(nc(df_hf[exp_col]).mean() or 0, color='#ef4444', linestyle='--', label=f"Avg: {nc(df_hf[exp_col]).mean() or 0:.1f} yrs")
                ax.set_title("Experience Distribution", fontweight='bold')
                ax.set_xlabel("Years of Experience")
                ax.legend()
                plt.tight_layout(); st.pyplot(fig); plt.close()
        if salary_col and exp_col:
            sh("📈","Salary vs Experience")
            fig, ax = plt.subplots(figsize=(10,4))
            clean_chart(fig, ax)
            ax.scatter(df_hf[exp_col], df_hf[salary_col], alpha=0.5, color='#6366f1', s=40)
            ax.set_xlabel("Years of Experience")
            ax.set_ylabel("Salary ($)")
            ax.set_title("Salary vs Experience Correlation", fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab6:
        hr_alerts = []
        if overtime_col and (df_hf[overtime_col] > 20).sum() > 0:
            hr_alerts.append(('red', f"🔴 <b>{(df_hf[overtime_col]>20).sum()} employees</b> working 20+ overtime hours — burnout risk"))
        if len(high_risk) > 0:
            hr_alerts.append(('red', f"🔴 <b>{len(high_risk)} employees</b> at HIGH attrition risk — act now"))
        if salary_col and dept_col:
            underpaid = (df_hf[salary_col] < df_hf.groupby(dept_col)[salary_col].transform('mean') * 0.8).sum()
            if underpaid > 0:
                hr_alerts.append(('amber', f"⚠️ <b>{underpaid} employees</b> earn 20%+ below department average — salary review needed"))
        if perf_col and (df_hf[perf_col] < df_hf[perf_col].quantile(0.2)).sum() > 0:
            hr_alerts.append(('amber', f"⚠️ <b>{(df_hf[perf_col] < df_hf[perf_col].quantile(0.2)).sum()} employees</b> in bottom 20% performance — coaching needed"))

        if not hr_alerts:
            abox("✅ HR looks healthy — no critical alerts detected.", "green")
        else:
            for kind, msg in hr_alerts:
                abox(msg, kind)

    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# MODULE: FINANCE & ACCOUNTING INTELLIGENCE
# ════════════════════════════════════════════════════════════════════════════
if module == "Finance & Accounting":
    df_fin = df.copy()

    # Detect columns
    date_col    = next((c for c in df_fin.columns if 'date' in c.lower() or 'month' in c.lower() or 'period' in c.lower()), None)
    type_col    = next((c for c in df_fin.columns if 'type' in c.lower() or 'category' in c.lower() or 'nature' in c.lower()), None)
    subcat_col  = next((c for c in df_fin.columns if 'sub' in c.lower() or 'item' in c.lower() or 'description' in c.lower()), None)
    amount_col  = next((c for c in df_fin.columns if 'amount' in c.lower() or 'value' in c.lower() or 'actual' in c.lower()), None)
    budget_col  = next((c for c in df_fin.columns if 'budget' in c.lower() or 'target' in c.lower() or 'plan' in c.lower()), None)
    dept_col    = next((c for c in df_fin.columns if 'dept' in c.lower() or 'department' in c.lower() or 'division' in c.lower()), None)
    vendor_col  = next((c for c in df_fin.columns if 'vendor' in c.lower() or 'supplier' in c.lower() or 'party' in c.lower()), None)

    if date_col:
        df_fin[date_col] = pd.to_datetime(df_fin[date_col], errors='coerce')

    show_clean_report(fixes, df_fin, df_raw)

    # Sidebar filters
    with st.sidebar:
        st.markdown("### 🔍 Filter Finance")
        df_ff = df_fin.copy()
        if type_col:
            types = ['All'] + sorted(df_fin[type_col].dropna().unique().tolist())
            st_ = st.selectbox("Type", types, key="fin_type")
            if st_ != 'All': df_ff = df_ff[df_ff[type_col] == st_]
        if dept_col:
            depts = ['All'] + sorted(df_fin[dept_col].dropna().unique().tolist())
            sd = st.selectbox("Department", depts, key="fin_dept")
            if sd != 'All': df_ff = df_ff[df_ff[dept_col] == sd]
        st.caption(f"**{len(df_ff)}** of **{len(df_fin)}** records")

    # Separate Revenue vs Expense
    if type_col and amount_col:
        rev_keywords  = ['revenue','income','sales','receipt','inflow']
        exp_keywords  = ['expense','cost','expenditure','payment','outflow','salary','rent']
        rev_mask = df_ff[type_col].astype(str).str.lower().str.contains('|'.join(rev_keywords), na=False)
        exp_mask = df_ff[type_col].astype(str).str.lower().str.contains('|'.join(exp_keywords), na=False)
        total_rev = df_ff[rev_mask][amount_col].sum() if rev_mask.sum() > 0 else 0
        total_exp = df_ff[exp_mask][amount_col].sum() if exp_mask.sum() > 0 else 0
    elif amount_col:
        total_rev = df_ff[df_ff[amount_col] > 0][amount_col].sum()
        total_exp = abs(df_ff[df_ff[amount_col] < 0][amount_col].sum())
    else:
        total_rev, total_exp = 0, 0

    net_profit    = total_rev - total_exp
    profit_margin = (net_profit / total_rev * 100) if total_rev > 0 else 0
    budget_var    = 0
    if budget_col and amount_col:
        budget_var = nc(df_ff[amount_col]).sum() - nc(df_ff[budget_col]).sum()

    kpi_row([
        (f"${total_rev:,.0f}", "Total Revenue", "green", "↑ Inflows"),
        (f"${total_exp:,.0f}", "Total Expenses", "red", "↓ Outflows"),
        (f"${net_profit:,.0f}", "Net Profit", "green" if net_profit >= 0 else "red", f"{profit_margin:.1f}% margin"),
        (f"{profit_margin:.1f}%", "Profit Margin", "green" if profit_margin >= 15 else "amber" if profit_margin >= 0 else "red", "Target: 15%+"),
        (f"${abs(budget_var):,.0f}", "Budget Variance", "green" if budget_var >= 0 else "red", "Under budget" if budget_var >= 0 else "Over budget") if budget_col else (f"${net_profit:,.0f}", "Surplus/Deficit", "green" if net_profit >= 0 else "red", None),
        (f"{len(df_ff):,}", "Total Records", "indigo", None),
    ])

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Revenue","💸 Expenses","⚖️ P&L Summary","📉 Budget vs Actual","🏢 Dept Analysis","🚨 Smart Alerts"
    ])

    with tab1:
        sh("📈","Revenue Analysis")
        if amount_col and type_col and rev_mask.sum() > 0:
            rev_df = df_ff[rev_mask]
            col_a, col_b = st.columns(2)
            with col_a:
                if date_col and rev_df[date_col].notna().sum() > 0:
                    monthly_rev = rev_df.groupby(rev_df[date_col].dt.to_period('M'))[amount_col].sum().reset_index()
                    monthly_rev.columns = ['Month','Revenue']
                    monthly_rev['Month'] = monthly_rev['Month'].astype(str)
                    fig, ax = plt.subplots(figsize=(7,4))
                    clean_chart(fig, ax)
                    ax.fill_between(range(len(monthly_rev)), monthly_rev['Revenue'], alpha=0.15, color='#6366f1')
                    ax.plot(range(len(monthly_rev)), monthly_rev['Revenue'], color='#6366f1', linewidth=2.5, marker='o', markersize=5)
                    ax.set_xticks(range(len(monthly_rev)))
                    ax.set_xticklabels(monthly_rev['Month'], rotation=45, ha='right', fontsize=8)
                    ax.set_title("Revenue Trend", fontweight='bold')
                    ax.set_ylabel("Revenue ($)")
                    plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                if subcat_col:
                    rev_by_cat = rev_df.groupby(subcat_col)[amount_col].sum().sort_values(ascending=False).head(8)
                    fig, ax = plt.subplots(figsize=(6,4))
                    clean_chart(fig, ax)
                    ax.pie(rev_by_cat.values, labels=rev_by_cat.index, autopct='%1.1f%%',
                           colors=['#6366f1','#818cf8','#a5b4fc','#c7d2fe','#e0e7ff','#4f46e5','#10b981','#34d399'],
                           wedgeprops=dict(edgecolor='white', linewidth=2))
                    ax.set_title("Revenue by Category", fontweight='bold')
                    plt.tight_layout(); st.pyplot(fig); plt.close()
        elif amount_col:
            # Fallback: show all positive amounts as revenue
            sh("ℹ️","Revenue Trend")
            if date_col:
                pos_df = df_ff[df_ff[amount_col] > 0]
                if len(pos_df) > 0:
                    monthly = pos_df.groupby(pos_df[date_col].dt.to_period('M'))[amount_col].sum().reset_index()
                    monthly.columns = ['Month','Amount']
                    monthly['Month'] = monthly['Month'].astype(str)
                    fig, ax = plt.subplots(figsize=(10,4))
                    clean_chart(fig, ax)
                    ax.fill_between(range(len(monthly)), monthly['Amount'], alpha=0.15, color='#6366f1')
                    ax.plot(range(len(monthly)), monthly['Amount'], color='#6366f1', linewidth=2.5, marker='o')
                    ax.set_xticks(range(len(monthly)))
                    ax.set_xticklabels(monthly['Month'], rotation=45, ha='right', fontsize=8)
                    ax.set_title("Revenue / Inflows Trend", fontweight='bold')
                    plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        sh("💸","Expense Breakdown")
        if amount_col and type_col and exp_mask.sum() > 0:
            exp_df = df_ff[exp_mask]
            col_a, col_b = st.columns(2)
            with col_a:
                if subcat_col:
                    exp_by_cat = exp_df.groupby(subcat_col)[amount_col].sum().sort_values(ascending=False).head(8)
                    fig, ax = plt.subplots(figsize=(6,4))
                    clean_chart(fig, ax)
                    ax.barh(exp_by_cat.index, exp_by_cat.values, color='#ef4444', alpha=0.8)
                    ax.invert_yaxis()
                    for i, val in enumerate(exp_by_cat.values):
                        ax.text(val, i, f'  ${val:,.0f}', va='center', fontsize=9)
                    ax.set_title("Top Expense Categories", fontweight='bold')
                    plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                if date_col and exp_df[date_col].notna().sum() > 0:
                    monthly_exp = exp_df.groupby(exp_df[date_col].dt.to_period('M'))[amount_col].sum().reset_index()
                    monthly_exp.columns = ['Month','Expense']
                    monthly_exp['Month'] = monthly_exp['Month'].astype(str)
                    fig, ax = plt.subplots(figsize=(7,4))
                    clean_chart(fig, ax)
                    ax.fill_between(range(len(monthly_exp)), monthly_exp['Expense'], alpha=0.12, color='#ef4444')
                    ax.plot(range(len(monthly_exp)), monthly_exp['Expense'], color='#ef4444', linewidth=2.5, marker='o', markersize=5)
                    ax.set_xticks(range(len(monthly_exp)))
                    ax.set_xticklabels(monthly_exp['Month'], rotation=45, ha='right', fontsize=8)
                    ax.set_title("Expense Trend", fontweight='bold')
                    plt.tight_layout(); st.pyplot(fig); plt.close()
            if vendor_col:
                sh("🏢","Top Vendors by Spend")
                vendor_spend = exp_df.groupby(vendor_col)[amount_col].sum().sort_values(ascending=False).head(10).reset_index()
                vendor_spend.columns = ['Vendor','Total Spend']
                vendor_spend['% of Total'] = (vendor_spend['Total Spend'] / vendor_spend['Total Spend'].sum() * 100).round(1)
                st.dataframe(vendor_spend.style.background_gradient(subset=['Total Spend'], cmap='Reds'), use_container_width=True)
        else:
            abox("ℹ️ Add a 'Type' column with 'Revenue'/'Expense' values for detailed breakdown.", "blue")

    with tab3:
        sh("⚖️","Profit & Loss Summary")
        pl_data = {
            'Metric': ['Total Revenue', 'Total Expenses', 'Gross Profit', 'Profit Margin %'],
            'Value': [f"${total_rev:,.0f}", f"${total_exp:,.0f}", f"${net_profit:,.0f}", f"{profit_margin:.1f}%"],
            'Status': ['✅', '📊', '✅' if net_profit >= 0 else '🔴', '✅' if profit_margin >= 15 else '⚠️' if profit_margin >= 0 else '🔴']
        }
        pl_df = pd.DataFrame(pl_data)
        col_a, col_b = st.columns(2)
        with col_a:
            st.dataframe(pl_df, use_container_width=True, hide_index=True)
            if net_profit >= 0:
                abox(f"✅ Business is <b>profitable</b> — ${net_profit:,.0f} net profit with {profit_margin:.1f}% margin.", "green")
            else:
                abox(f"🔴 Business is running at a <b>loss</b> of ${abs(net_profit):,.0f}. Review expense categories immediately.", "red")
        with col_b:
            if total_rev > 0 and total_exp > 0:
                fig, ax = plt.subplots(figsize=(5,5))
                clean_chart(fig, ax)
                labels = ['Revenue','Expenses']
                vals = [total_rev, total_exp]
                bar_c = ['#10b981','#ef4444']
                bars = ax.bar(labels, vals, color=bar_c, width=0.5, edgecolor='white', linewidth=2)
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'${val:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
                ax.set_title("Revenue vs Expenses", fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab4:
        if budget_col and amount_col:
            sh("📉","Budget vs Actual Analysis")
            df_ff['Variance'] = df_ff[amount_col] - df_ff[budget_col]
            df_ff['Variance %'] = (df_ff['Variance'] / df_ff[budget_col] * 100).round(1)
            total_budget  = nc(df_ff[budget_col]).sum()
            total_actual  = nc(df_ff[amount_col]).sum()
            total_variance = total_actual - total_budget

            kpi_row([
                (f"${total_budget:,.0f}", "Total Budget", "indigo", None),
                (f"${total_actual:,.0f}", "Total Actual", "indigo", None),
                (f"${abs(total_variance):,.0f}", "Total Variance", "green" if total_variance <= 0 else "red", "Under budget ✅" if total_variance <= 0 else "Over budget ⚠️"),
            ])

            if type_col:
                var_by_type = df_ff.groupby(type_col).agg({amount_col:'sum', budget_col:'sum'}).reset_index()
                var_by_type.columns = ['Category','Actual','Budget']
                var_by_type['Variance'] = var_by_type['Actual'] - var_by_type['Budget']
                var_by_type['Over Budget'] = var_by_type['Variance'] > 0
                fig, ax = plt.subplots(figsize=(10,4))
                clean_chart(fig, ax)
                x = range(len(var_by_type))
                w = 0.35
                ax.bar([i-w/2 for i in x], var_by_type['Budget'], w, label='Budget', color='#e0e7ff', edgecolor='white')
                ax.bar([i+w/2 for i in x], var_by_type['Actual'], w, label='Actual',
                       color=['#ef4444' if v else '#10b981' for v in var_by_type['Over Budget']], edgecolor='white')
                ax.set_xticks(list(x))
                ax.set_xticklabels(var_by_type['Category'], rotation=30, ha='right', fontsize=9)
                ax.legend(); ax.set_title("Budget vs Actual by Category", fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()

                over = var_by_type[var_by_type['Over Budget']]
                if len(over) > 0:
                    abox(f"⚠️ <b>{len(over)} categories</b> are over budget. Review spending immediately.", "amber")
                    st.dataframe(over.style.background_gradient(subset=['Variance'], cmap='Reds'), use_container_width=True)
        else:
            abox("ℹ️ Add a 'Budget' column alongside 'Amount' for variance analysis.", "blue")

    with tab5:
        if dept_col and amount_col:
            sh("🏢","Department Cost Analysis")
            dept_spend = df_ff.groupby(dept_col)[amount_col].sum().sort_values(ascending=False).reset_index()
            dept_spend.columns = ['Department','Total Amount']
            dept_spend['% of Total'] = (dept_spend['Total Amount']/dept_spend['Total Amount'].sum()*100).round(1)

            col_a, col_b = st.columns(2)
            with col_a:
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                ax.barh(dept_spend['Department'], dept_spend['Total Amount'], color='#6366f1')
                ax.invert_yaxis()
                for i, val in enumerate(dept_spend['Total Amount']):
                    ax.text(val, i, f'  ${val:,.0f}', va='center', fontsize=9)
                ax.set_title("Spend by Department", fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                st.dataframe(dept_spend.style.background_gradient(subset=['Total Amount'], cmap='Blues'), use_container_width=True)
        else:
            abox("ℹ️ Add a 'Department' column for department-wise cost analysis.", "blue")

    with tab6:
        fin_alerts = []
        if profit_margin < 0:
            fin_alerts.append(('red', f"🔴 Business is running at a <b>loss of ${abs(net_profit):,.0f}</b> — immediate cost review required"))
        elif profit_margin < 10:
            fin_alerts.append(('amber', f"⚠️ Profit margin is only <b>{profit_margin:.1f}%</b> — below healthy 15% threshold"))
        if budget_col and amount_col:
            over_budget_total = (df_ff[amount_col] - df_ff[budget_col]).sum()
            if over_budget_total > 0:
                fin_alerts.append(('amber', f"⚠️ Total spending is <b>${over_budget_total:,.0f} over budget</b> — review department spending"))
        if total_exp > total_rev * 0.85:
            fin_alerts.append(('amber', f"⚠️ Expenses are <b>{total_exp/total_rev*100:.0f}% of revenue</b> — cost discipline needed"))

        if not fin_alerts:
            abox("✅ Finance looks healthy — no critical alerts detected.", "green")
        else:
            for kind, msg in fin_alerts:
                abox(msg, kind)

    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# MODULE: SALES INTELLIGENCE
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# MODULE: RETAIL & E-COMMERCE INTELLIGENCE
# ════════════════════════════════════════════════════════════════════════════
if module == "Retail & E-commerce":
    df_ret = df.copy()
    show_clean_report(fixes, df_ret, df_raw)

    # Detect columns
    date_col   = next((c for c in df_ret.columns if 'date' in c.lower()), None)
    prod_col   = next((c for c in df_ret.columns if 'product' in c.lower()), None)
    cat_col    = next((c for c in df_ret.columns if 'category' in c.lower()), None)
    chan_col   = next((c for c in df_ret.columns if 'channel' in c.lower()), None)
    rev_col    = next((c for c in df_ret.columns if 'revenue' in c.lower()), None)
    qty_col    = next((c for c in df_ret.columns if 'quantity' in c.lower() or 'qty' in c.lower()), None)
    price_col  = next((c for c in df_ret.columns if 'price' in c.lower() and 'cost' not in c.lower()), None)
    disc_col   = next((c for c in df_ret.columns if 'discount' in c.lower()), None)
    ctype_col  = next((c for c in df_ret.columns if 'customer type' in c.lower() or 'customer' in c.lower()), None)
    return_col = next((c for c in df_ret.columns if 'return' in c.lower()), None)
    rating_col = next((c for c in df_ret.columns if 'rating' in c.lower()), None)
    city_col   = next((c for c in df_ret.columns if 'city' in c.lower() or 'region' in c.lower()), None)
    pay_col    = next((c for c in df_ret.columns if 'payment' in c.lower()), None)
    deliv_col  = next((c for c in df_ret.columns if 'delivery' in c.lower() and 'day' in c.lower()), None)

    if date_col: df_ret[date_col] = pd.to_datetime(df_ret[date_col], dayfirst=True, errors='coerce')

    # Core metrics
    total_rev    = nc(df_ret[rev_col]).sum() if rev_col else 0
    total_orders = len(df_ret)
    aov          = total_rev / total_orders if total_orders > 0 else 0
    return_rate  = (df_ret[return_col].astype(str).str.upper() == 'YES').mean() * 100 if return_col else 0
    avg_rating   = nc(df_ret[rating_col]).mean() or 0 if rating_col else 0
    new_cust_pct = (df_ret[ctype_col].astype(str).str.title() == 'New').mean() * 100 if ctype_col else 0

    kpi_row([
        (f"₹{total_rev:,.0f}", "Total Revenue", "indigo", None),
        (f"{total_orders:,}", "Total Orders", "indigo", None),
        (f"₹{aov:,.0f}", "Avg Order Value", "green", "Per order"),
        (f"{return_rate:.1f}%", "Return Rate", "red" if return_rate > 15 else "amber" if return_rate > 8 else "green", "⚠️ High" if return_rate > 15 else "✅ Healthy"),
        (f"{avg_rating:.1f}★", "Avg Rating", "green" if avg_rating >= 4 else "amber", f"Out of 5.0"),
        (f"{new_cust_pct:.0f}%", "New Customers", "indigo", "of total orders"),
    ])

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Sales Overview", "📦 Product Performance", "🛒 Channel Analysis",
        "👥 Customer Insights", "🚚 Delivery & Returns", "🚨 Smart Alerts"
    ])

    with tab1:
        sh("📈", "Revenue Trend")
        if date_col and rev_col:
            col_a, col_b = st.columns(2)
            with col_a:
                monthly = df_ret.groupby(df_ret[date_col].dt.to_period('M'))[rev_col].sum().reset_index()
                monthly.columns = ['Month','Revenue']
                monthly['Month'] = monthly['Month'].astype(str)
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                ax.fill_between(range(len(monthly)), monthly['Revenue'], alpha=0.12, color='#6366f1')
                ax.plot(range(len(monthly)), monthly['Revenue'], color='#6366f1', linewidth=2.5, marker='o', markersize=5)
                ax.set_xticks(range(len(monthly)))
                ax.set_xticklabels(monthly['Month'], rotation=45, ha='right', fontsize=8)
                ax.set_title("Monthly Revenue Trend", fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                if cat_col:
                    cat_rev = df_ret.groupby(cat_col)[rev_col].sum().sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(6,4))
                    clean_chart(fig, ax)
                    ax.pie(cat_rev.values, labels=cat_rev.index, autopct='%1.1f%%',
                           colors=['#6366f1','#818cf8','#a5b4fc','#c7d2fe','#e0e7ff','#4f46e5'],
                           wedgeprops=dict(edgecolor='white', linewidth=2))
                    ax.set_title("Revenue by Category", fontweight='bold')
                    plt.tight_layout(); st.pyplot(fig); plt.close()

        if disc_col and rev_col:
            sh("💸","Discount Impact Analysis")
            df_ret['Discount Band'] = pd.cut(df_ret[disc_col], bins=[-1,0,10,20,30,100], labels=['No discount','1-10%','11-20%','21-30%','30%+'])
            disc_impact = df_ret.groupby('Discount Band').agg({rev_col:'sum', qty_col:'sum'} if qty_col else {rev_col:'sum'}).reset_index()
            col_c, col_d = st.columns(2)
            with col_c:
                fig, ax = plt.subplots(figsize=(6,3))
                clean_chart(fig, ax)
                ax.bar(disc_impact['Discount Band'].astype(str), disc_impact[rev_col], color='#6366f1', alpha=0.85, width=0.5)
                ax.set_title("Revenue by Discount Level", fontweight='bold')
                plt.xticks(rotation=20, ha='right', fontsize=9)
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_d:
                st.dataframe(disc_impact, use_container_width=True)

    with tab2:
        if prod_col and rev_col:
            col_a, col_b = st.columns(2)
            with col_a:
                sh("🏆","Top 10 Products by Revenue")
                top_p = df_ret.groupby(prod_col)[rev_col].sum().sort_values(ascending=False).head(10)
                fig, ax = plt.subplots(figsize=(6,5))
                clean_chart(fig, ax)
                top_p.plot(kind='barh', ax=ax, color='#6366f1')
                ax.invert_yaxis()
                for i, val in enumerate(top_p.values):
                    ax.text(val, i, f' ₹{val:,.0f}', va='center', fontsize=8)
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                sh("⚠️","Bottom 10 Products")
                bot_p = df_ret.groupby(prod_col)[rev_col].sum().sort_values().head(10)
                fig, ax = plt.subplots(figsize=(6,5))
                clean_chart(fig, ax)
                bot_p.plot(kind='barh', ax=ax, color='#ef4444', alpha=0.8)
                ax.invert_yaxis()
                plt.tight_layout(); st.pyplot(fig); plt.close()

            if return_col:
                sh("↩️","Return Rate by Product")
                ret_by_prod = df_ret.groupby(prod_col).agg(
                    Total_Orders=(rev_col,'count'),
                    Returns=(return_col, lambda x: (x.str.upper()=='YES').sum())
                ).reset_index()
                ret_by_prod['Return Rate %'] = (ret_by_prod['Returns']/ret_by_prod['Total_Orders']*100).round(1)
                high_return = ret_by_prod[ret_by_prod['Return Rate %'] > 15].sort_values('Return Rate %', ascending=False)
                if len(high_return) > 0:
                    abox(f"⚠️ <b>{len(high_return)} products</b> have return rates above 15% — review quality or product descriptions.", "amber")
                    st.dataframe(high_return, use_container_width=True)
                else:
                    abox("✅ Return rates are healthy across all products.", "green")

    with tab3:
        if chan_col and rev_col:
            sh("🛒","Channel Performance")
            col_a, col_b = st.columns(2)
            with col_a:
                chan_rev = df_ret.groupby(chan_col)[rev_col].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                bars = ax.bar(chan_rev.index, chan_rev.values, color=['#6366f1','#818cf8','#a5b4fc','#c7d2fe','#4f46e5'], width=0.5)
                ax.set_title("Revenue by Channel", fontweight='bold')
                plt.xticks(rotation=20, ha='right', fontsize=9)
                for bar, val in zip(bars, chan_rev.values):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'₹{val:,.0f}', ha='center', va='bottom', fontsize=8)
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                chan_summary = df_ret.groupby(chan_col).agg(
                    Orders=(rev_col,'count'),
                    Revenue=(rev_col,'sum'),
                ).reset_index()
                if qty_col: chan_summary['Avg Order Value'] = (chan_summary['Revenue']/chan_summary['Orders']).round(0)
                chan_summary['Revenue %'] = (chan_summary['Revenue']/chan_summary['Revenue'].sum()*100).round(1)
                chan_summary['Revenue'] = chan_summary['Revenue'].apply(lambda x: f"₹{x:,.0f}")
                st.dataframe(chan_summary, use_container_width=True)

            if pay_col:
                sh("💳","Payment Method Analysis")
                pay_rev = df_ret.groupby(pay_col)[rev_col].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(8,3))
                clean_chart(fig, ax)
                ax.pie(pay_rev.values, labels=pay_rev.index, autopct='%1.1f%%',
                       colors=['#6366f1','#818cf8','#a5b4fc','#c7d2fe','#e0e7ff'],
                       wedgeprops=dict(edgecolor='white', linewidth=2), startangle=90)
                plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab4:
        if ctype_col and rev_col:
            sh("👥","New vs Returning Customers")
            col_a, col_b = st.columns(2)
            with col_a:
                ctype_rev = df_ret.groupby(ctype_col)[rev_col].sum()
                fig, ax = plt.subplots(figsize=(5,5))
                clean_chart(fig, ax)
                ax.pie(ctype_rev.values, labels=ctype_rev.index, autopct='%1.1f%%',
                       colors=['#6366f1','#10b981'], wedgeprops=dict(edgecolor='white', linewidth=3), startangle=90)
                ax.set_title("Revenue Split", fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                ctype_stats = df_ret.groupby(ctype_col).agg(
                    Orders=(rev_col,'count'), Revenue=(rev_col,'sum')
                ).reset_index()
                ctype_stats['AOV'] = (ctype_stats['Revenue']/ctype_stats['Orders']).round(0).apply(lambda x: f"₹{x:,.0f}")
                ctype_stats['Revenue'] = ctype_stats['Revenue'].apply(lambda x: f"₹{x:,.0f}")
                st.dataframe(ctype_stats, use_container_width=True, hide_index=True)
                abox("💡 <b>Insight:</b> Returning customers typically have 30-40% higher AOV. Focus retention campaigns on converting new buyers to repeat buyers.", "blue")

        if city_col and rev_col:
            sh("🌍","Revenue by City")
            city_rev = df_ret.groupby(city_col)[rev_col].sum().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(10,3))
            clean_chart(fig, ax)
            bars = ax.bar(city_rev.index, city_rev.values, color='#6366f1', alpha=0.85, width=0.6)
            ax.set_title("Top Cities by Revenue", fontweight='bold')
            for bar, val in zip(bars, city_rev.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'₹{val:,.0f}', ha='center', va='bottom', fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab5:
        col_a, col_b = st.columns(2)
        with col_a:
            if return_col:
                sh("↩️","Returns Analysis")
                ret_counts = df_ret[return_col].astype(str).str.title().value_counts()
                fig, ax = plt.subplots(figsize=(5,4))
                clean_chart(fig, ax)
                ax.pie(ret_counts.values, labels=ret_counts.index, autopct='%1.1f%%',
                       colors=['#10b981','#ef4444'], wedgeprops=dict(edgecolor='white', linewidth=3))
                ax.set_title(f"Return Rate: {return_rate:.1f}%", fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
        with col_b:
            if deliv_col:
                sh("🚚","Delivery Time Distribution")
                fig, ax = plt.subplots(figsize=(5,4))
                clean_chart(fig, ax)
                ax.hist(df_ret[deliv_col].dropna(), bins=10, color='#818cf8', edgecolor='white', linewidth=1.5)
                ax.axvline(nc(df_ret[deliv_col]).mean() or 0, color='#ef4444', linestyle='--', label=f"Avg: {nc(df_ret[deliv_col]).mean() or 0:.1f} days")
                ax.set_title("Delivery Days Distribution", fontweight='bold')
                ax.set_xlabel("Days")
                ax.legend()
                plt.tight_layout(); st.pyplot(fig); plt.close()

        if rating_col:
            sh("⭐","Customer Ratings")
            col_c, col_d = st.columns(2)
            with col_c:
                fig, ax = plt.subplots(figsize=(5,3))
                clean_chart(fig, ax)
                ax.hist(df_ret[rating_col].dropna(), bins=10, color='#fbbf24', edgecolor='white', linewidth=1.5)
                ax.axvline(avg_rating, color='#ef4444', linestyle='--', label=f"Avg: {avg_rating:.1f}")
                ax.set_title("Rating Distribution", fontweight='bold')
                ax.legend()
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_d:
                low_rated = (df_ret[rating_col] < 3).sum()
                if low_rated > 0:
                    abox(f"⚠️ <b>{low_rated} orders</b> rated below 3 stars — investigate for quality issues.", "amber")
                high_rated = (df_ret[rating_col] >= 4.5).sum()
                abox(f"✅ <b>{high_rated} orders</b> rated 4.5+ stars — excellent customer satisfaction!", "green")

    with tab6:
        r_alerts = []
        if return_rate > 15: r_alerts.append(('red', f"🔴 Return rate is <b>{return_rate:.1f}%</b> — above 15% threshold. Review product quality and descriptions."))
        if avg_rating < 3.5 and avg_rating > 0: r_alerts.append(('red', f"🔴 Average rating is <b>{avg_rating:.1f}</b> — below 3.5. Immediate customer experience review needed."))
        if aov < 500 and total_orders > 0: r_alerts.append(('amber', f"⚠️ Average order value is only <b>₹{aov:,.0f}</b> — consider bundling or upselling strategies."))
        if new_cust_pct > 70: r_alerts.append(('amber', f"⚠️ <b>{new_cust_pct:.0f}%</b> orders from new customers — low retention. Build loyalty programme."))
        if not r_alerts: abox("✅ All retail metrics look healthy — no critical alerts.", "green")
        else:
            for k, m in r_alerts: abox(m, k)
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# MODULE: LOGISTICS & SUPPLY CHAIN INTELLIGENCE
# ════════════════════════════════════════════════════════════════════════════
if module == "Logistics":
    df_log = df.copy()
    show_clean_report(fixes, df_log, df_raw)

    date_col    = next((c for c in df_log.columns if 'date' in c.lower()), None)
    route_col   = next((c for c in df_log.columns if 'route' in c.lower()), None)
    vehicle_col = next((c for c in df_log.columns if 'vehicle' in c.lower()), None)
    driver_col  = next((c for c in df_log.columns if 'driver' in c.lower()), None)
    cargo_col   = next((c for c in df_log.columns if 'cargo' in c.lower() or 'type' in c.lower()), None)
    weight_col  = next((c for c in df_log.columns if 'weight' in c.lower()), None)
    plan_col    = next((c for c in df_log.columns if 'planned' in c.lower()), None)
    actual_col  = next((c for c in df_log.columns if 'actual' in c.lower() and 'day' in c.lower()), None)
    ontime_col  = next((c for c in df_log.columns if 'on time' in c.lower() or 'ontime' in c.lower()), None)
    cost_col    = next((c for c in df_log.columns if 'delivery cost' in c.lower() or ('cost' in c.lower() and 'budget' not in c.lower())), None)
    budget_col  = next((c for c in df_log.columns if 'budget' in c.lower()), None)
    damage_col  = next((c for c in df_log.columns if 'damage' in c.lower()), None)
    rating_col  = next((c for c in df_log.columns if 'rating' in c.lower()), None)
    dist_col    = next((c for c in df_log.columns if 'distance' in c.lower()), None)

    if date_col: df_log[date_col] = pd.to_datetime(df_log[date_col], dayfirst=True, errors='coerce')

    total_ship   = len(df_log)
    ontime_rate  = (df_log[ontime_col].astype(str).str.upper() == 'YES').mean() * 100 if ontime_col else 0
    total_cost   = nc(df_log[cost_col]).sum() if cost_col else 0
    damage_rate  = (df_log[damage_col].astype(str).str.upper() == 'YES').mean() * 100 if damage_col else 0
    avg_rating   = nc(df_log[rating_col]).mean() or 0 if rating_col else 0
    cost_per_km  = (nc(df_log[cost_col]).sum() / nc(df_log[dist_col]).sum()) if cost_col and dist_col and nc(df_log[dist_col]).sum() > 0 else 0

    kpi_row([
        (f"{total_ship:,}", "Total Shipments", "indigo", None),
        (f"{ontime_rate:.1f}%", "On-Time Rate", "green" if ontime_rate >= 85 else "amber" if ontime_rate >= 70 else "red", "Target: 85%+"),
        (f"₹{total_cost:,.0f}", "Total Logistics Cost", "indigo", None),
        (f"{damage_rate:.1f}%", "Damage Rate", "green" if damage_rate < 3 else "amber" if damage_rate < 7 else "red", "⚠️ High" if damage_rate > 5 else "✅ OK"),
        (f"{avg_rating:.1f}★", "Avg Rating", "green" if avg_rating >= 4 else "amber", "Out of 5.0"),
        (f"₹{cost_per_km:.0f}", "Cost per KM", "indigo", "Efficiency metric"),
    ])

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚚 Delivery Performance", "💰 Cost Analysis", "🚛 Fleet & Drivers",
        "📦 Cargo Analysis", "🚨 Smart Alerts"
    ])

    with tab1:
        sh("🚚","On-Time Delivery Analysis")
        col_a, col_b = st.columns(2)
        with col_a:
            if ontime_col:
                ot_counts = df_log[ontime_col].astype(str).str.title().value_counts()
                fig, ax = plt.subplots(figsize=(5,5))
                clean_chart(fig, ax)
                ax.pie(ot_counts.values, labels=ot_counts.index, autopct='%1.1f%%',
                       colors=['#10b981','#ef4444'], wedgeprops=dict(edgecolor='white', linewidth=3), startangle=90)
                ax.set_title(f"On-Time Rate: {ontime_rate:.1f}%", fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
        with col_b:
            if route_col and ontime_col:
                route_ot = df_log.groupby(route_col).apply(lambda x: (x[ontime_col].astype(str).str.upper()=='YES').mean()*100).sort_values()
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                bar_c = ['#ef4444' if v < 70 else '#f59e0b' if v < 85 else '#10b981' for v in route_ot.values]
                ax.barh(route_ot.index, route_ot.values, color=bar_c)
                ax.axvline(85, color='#6366f1', linestyle='--', linewidth=1.5, label='Target 85%')
                ax.set_title("On-Time Rate by Route", fontweight='bold')
                ax.set_xlabel("On-Time %")
                ax.legend(fontsize=9)
                for i, val in enumerate(route_ot.values):
                    ax.text(val+0.5, i, f'{val:.0f}%', va='center', fontsize=9)
                plt.tight_layout(); st.pyplot(fig); plt.close()

        if plan_col and actual_col:
            sh("⏱️","Delay Analysis")
            df_log['Delay Days'] = df_log[actual_col] - df_log[plan_col]
            delayed = df_log[df_log['Delay Days'] > 0]
            kpi_row([
                (f"{len(delayed)}", "Delayed Shipments", "red" if len(delayed)/total_ship > 0.2 else "amber", f"{len(delayed)/total_ship*100:.1f}% of total"),
                (f"{df_log['Delay Days'].mean():.1f}", "Avg Delay Days", "amber", "When delayed"),
                (f"{df_log['Delay Days'].max():.0f}", "Max Delay Days", "red", "Worst case"),
            ])

    with tab2:
        sh("💰","Cost Analysis")
        if cost_col:
            col_a, col_b = st.columns(2)
            with col_a:
                if route_col:
                    route_cost = df_log.groupby(route_col)[cost_col].sum().sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(6,4))
                    clean_chart(fig, ax)
                    ax.barh(route_cost.index, route_cost.values, color='#6366f1')
                    ax.invert_yaxis()
                    ax.set_title("Total Cost by Route", fontweight='bold')
                    for i, val in enumerate(route_cost.values):
                        ax.text(val, i, f'  ₹{val:,.0f}', va='center', fontsize=9)
                    plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                if budget_col:
                    df_log['Variance'] = df_log[cost_col] - df_log[budget_col]
                    over_budget = (df_log['Variance'] > 0).sum()
                    total_variance = df_log['Variance'].sum()
                    kpi_row([
                        (f"{over_budget}", "Over Budget Trips", "red" if over_budget > total_ship*0.3 else "amber", f"{over_budget/total_ship*100:.0f}% of trips"),
                        (f"₹{abs(total_variance):,.0f}", "Total Variance", "red" if total_variance > 0 else "green", "Over budget" if total_variance > 0 else "Under budget"),
                    ])
                    fig, ax = plt.subplots(figsize=(6,4))
                    clean_chart(fig, ax)
                    ax.hist(df_log['Variance'], bins=20, color=['#ef4444' if v > 0 else '#10b981' for v in df_log['Variance']], edgecolor='white')
                    ax.axvline(0, color='#6b7280', linestyle='--', linewidth=1.5)
                    ax.set_title("Cost Variance Distribution", fontweight='bold')
                    ax.set_xlabel("₹ Over/Under Budget")
                    plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            if vehicle_col and cost_col:
                sh("🚛","Vehicle Utilisation & Cost")
                veh_stats = df_log.groupby(vehicle_col).agg(
                    Trips=(cost_col,'count'), Total_Cost=(cost_col,'sum')
                ).reset_index()
                if ontime_col:
                    veh_ot = df_log.groupby(vehicle_col).apply(lambda x: (x[ontime_col].astype(str).str.upper()=='YES').mean()*100).reset_index()
                    veh_ot.columns = [vehicle_col,'On_Time_%']
                    veh_stats = veh_stats.merge(veh_ot, on=vehicle_col)
                st.dataframe(veh_stats.sort_values('Total_Cost', ascending=False).reset_index(drop=True), use_container_width=True)
        with col_b:
            if driver_col and ontime_col:
                sh("👤","Driver Performance")
                drv_stats = df_log.groupby(driver_col).apply(lambda x: (x[ontime_col].astype(str).str.upper()=='YES').mean()*100).sort_values(ascending=False).reset_index()
                drv_stats.columns = ['Driver','On-Time %']
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                bar_c = ['#10b981' if v >= 85 else '#f59e0b' if v >= 70 else '#ef4444' for v in drv_stats['On-Time %']]
                ax.barh(drv_stats['Driver'], drv_stats['On-Time %'], color=bar_c)
                ax.axvline(85, color='#6366f1', linestyle='--', linewidth=1.5, label='Target')
                ax.set_title("Driver On-Time Performance", fontweight='bold')
                ax.legend(fontsize=9)
                plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab4:
        if cargo_col and cost_col:
            sh("📦","Cargo Type Analysis")
            cargo_stats = df_log.groupby(cargo_col).agg(
                Shipments=(cost_col,'count'), Total_Cost=(cost_col,'sum')
            ).reset_index()
            if damage_col:
                cargo_dmg = df_log.groupby(cargo_col).apply(lambda x: (x[damage_col].astype(str).str.upper()=='YES').mean()*100).reset_index()
                cargo_dmg.columns = [cargo_col,'Damage_Rate_%']
                cargo_stats = cargo_stats.merge(cargo_dmg, on=cargo_col)
            fig, ax = plt.subplots(figsize=(8,4))
            clean_chart(fig, ax)
            ax.bar(cargo_stats[cargo_col], cargo_stats['Total_Cost'], color='#818cf8', width=0.5)
            ax.set_title("Cost by Cargo Type", fontweight='bold')
            plt.xticks(rotation=20, ha='right')
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.dataframe(cargo_stats.reset_index(drop=True), use_container_width=True)

    with tab5:
        log_alerts = []
        if ontime_rate < 70: log_alerts.append(('red', f"🔴 On-time delivery rate is only <b>{ontime_rate:.1f}%</b> — critical. Review routes and driver performance."))
        elif ontime_rate < 85: log_alerts.append(('amber', f"⚠️ On-time rate <b>{ontime_rate:.1f}%</b> is below 85% target. Needs improvement."))
        if damage_rate > 5: log_alerts.append(('red', f"🔴 Damage rate <b>{damage_rate:.1f}%</b> is above 5% — review packaging and handling."))
        if cost_col and budget_col:
            over_pct = (df_log[cost_col] > df_log[budget_col]).mean() * 100
            if over_pct > 30: log_alerts.append(('amber', f"⚠️ <b>{over_pct:.0f}%</b> of shipments are over budget — cost control needed."))
        if not log_alerts: abox("✅ Logistics operations look healthy — no critical alerts.", "green")
        else:
            for k, m in log_alerts: abox(m, k)
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# MODULE: RESTAURANT & FOOD SERVICE INTELLIGENCE
# ════════════════════════════════════════════════════════════════════════════
if module == "Restaurant":
    df_rest = df.copy()
    show_clean_report(fixes, df_rest, df_raw)

    date_col    = next((c for c in df_rest.columns if 'date' in c.lower()), None)
    item_col    = next((c for c in df_rest.columns if 'menu' in c.lower() or 'item' in c.lower()), None)
    cat_col     = next((c for c in df_rest.columns if 'category' in c.lower()), None)
    chan_col    = next((c for c in df_rest.columns if 'channel' in c.lower()), None)
    qty_col     = next((c for c in df_rest.columns if 'quantity' in c.lower() or 'qty' in c.lower()), None)
    sell_col    = next((c for c in df_rest.columns if 'selling' in c.lower()), None)
    cost_col    = next((c for c in df_rest.columns if 'cost' in c.lower()), None)
    rev_col     = next((c for c in df_rest.columns if 'revenue' in c.lower()), None)
    staff_col   = next((c for c in df_rest.columns if 'staff' in c.lower()), None)
    shift_col   = next((c for c in df_rest.columns if 'shift' in c.lower()), None)
    tables_col  = next((c for c in df_rest.columns if 'tables occupied' in c.lower()), None)
    total_t_col = next((c for c in df_rest.columns if 'total tables' in c.lower()), None)
    rating_col  = next((c for c in df_rest.columns if 'rating' in c.lower()), None)
    waste_col   = next((c for c in df_rest.columns if 'wastage' in c.lower() or 'waste' in c.lower()), None)

    if date_col: df_rest[date_col] = pd.to_datetime(df_rest[date_col], dayfirst=True, errors='coerce')

    total_rev     = nc(df_rest[rev_col]).sum() if rev_col else 0
    total_orders  = len(df_rest)
    avg_rating    = nc(df_rest[rating_col]).mean() or 0 if rating_col else 0
    occupancy_rate= (df_rest[tables_col] / df_rest[total_t_col]).mean() * 100 if tables_col and total_t_col else 0
    if sell_col and cost_col:
        df_rest['Food Cost %'] = (df_rest[cost_col] / df_rest[sell_col] * 100).round(1)
        avg_food_cost = df_rest['Food Cost %'].mean()
    else:
        avg_food_cost = 0
    total_waste = nc(df_rest[waste_col]).sum() if waste_col else 0

    kpi_row([
        (f"₹{total_rev:,.0f}", "Total Revenue", "indigo", None),
        (f"{total_orders:,}", "Total Orders", "indigo", None),
        (f"{occupancy_rate:.1f}%", "Table Occupancy", "green" if occupancy_rate >= 70 else "amber", "Target: 70%+"),
        (f"{avg_food_cost:.1f}%", "Avg Food Cost %", "green" if avg_food_cost <= 35 else "amber" if avg_food_cost <= 45 else "red", "Target: <35%"),
        (f"{avg_rating:.1f}★", "Avg Rating", "green" if avg_rating >= 4 else "amber", "Out of 5.0"),
        (f"{total_waste:,.0f}", "Total Wastage Units", "amber" if total_waste > 0 else "green", "Review waste"),
    ])

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🍽️ Menu Performance", "💰 Food Cost Analysis", "📊 Shift & Channel",
        "👨‍🍳 Staff Performance", "🚨 Smart Alerts"
    ])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            if item_col and rev_col:
                sh("🏆","Top 10 Best-Selling Items")
                top_items = df_rest.groupby(item_col)[rev_col].sum().sort_values(ascending=False).head(10)
                fig, ax = plt.subplots(figsize=(6,5))
                clean_chart(fig, ax)
                top_items.plot(kind='barh', ax=ax, color='#6366f1')
                ax.invert_yaxis()
                for i, val in enumerate(top_items.values):
                    ax.text(val, i, f' ₹{val:,.0f}', va='center', fontsize=8)
                plt.tight_layout(); st.pyplot(fig); plt.close()
        with col_b:
            if cat_col and rev_col:
                sh("📊","Revenue by Category")
                cat_rev = df_rest.groupby(cat_col)[rev_col].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(5,5))
                clean_chart(fig, ax)
                ax.pie(cat_rev.values, labels=cat_rev.index, autopct='%1.1f%%',
                       colors=['#6366f1','#818cf8','#a5b4fc','#c7d2fe'],
                       wedgeprops=dict(edgecolor='white', linewidth=2), startangle=90)
                ax.set_title("Sales by Category", fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()

        if item_col and qty_col:
            sh("📉","Slow-Moving Menu Items")
            slow_items = df_rest.groupby(item_col)[qty_col].sum().sort_values().head(8).reset_index()
            slow_items.columns = ['Menu Item','Units Sold']
            abox(f"⚠️ Consider removing or repricing these <b>slow-moving items</b> to reduce waste and simplify operations.", "amber")
            st.dataframe(slow_items, use_container_width=True, hide_index=True)

    with tab2:
        if 'Food Cost %' in df_rest.columns:
            sh("💰","Food Cost % by Category")
            col_a, col_b = st.columns(2)
            with col_a:
                if cat_col:
                    cat_fc = df_rest.groupby(cat_col)['Food Cost %'].mean().sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(6,4))
                    clean_chart(fig, ax)
                    bar_c = ['#ef4444' if v > 45 else '#f59e0b' if v > 35 else '#10b981' for v in cat_fc.values]
                    ax.barh(cat_fc.index, cat_fc.values, color=bar_c)
                    ax.axvline(35, color='#6366f1', linestyle='--', linewidth=1.5, label='Target 35%')
                    ax.set_title("Avg Food Cost % by Category", fontweight='bold')
                    ax.set_xlabel("Food Cost %")
                    ax.legend(fontsize=9)
                    for i, val in enumerate(cat_fc.values):
                        ax.text(val+0.2, i, f'{val:.1f}%', va='center', fontsize=9)
                    plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                if item_col:
                    high_cost = df_rest.groupby(item_col)['Food Cost %'].mean().sort_values(ascending=False).head(10).reset_index()
                    high_cost.columns = ['Item','Food Cost %']
                    abox(f"⚠️ Items above 35% food cost are eroding your margins — review pricing or portion size.", "amber")
                    st.dataframe(high_cost.style.background_gradient(subset=['Food Cost %'], cmap='Reds'), use_container_width=True)

        if waste_col and item_col:
            sh("🗑️","Wastage Analysis")
            waste_by_item = df_rest.groupby(item_col)[waste_col].sum().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(10,3))
            clean_chart(fig, ax)
            ax.bar(waste_by_item.index, waste_by_item.values, color='#ef4444', alpha=0.8, width=0.6)
            ax.set_title("Top Items by Wastage Units", fontweight='bold')
            plt.xticks(rotation=30, ha='right', fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            if shift_col and rev_col:
                sh("⏰","Revenue by Shift")
                shift_rev = df_rest.groupby(shift_col)[rev_col].sum().reindex(['Morning','Afternoon','Evening','Night'], fill_value=0)
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                shift_colors = ['#f59e0b','#6366f1','#10b981','#1e1b4b']
                bars = ax.bar(shift_rev.index, shift_rev.values, color=shift_colors, width=0.5)
                ax.set_title("Revenue by Shift", fontweight='bold')
                for bar, val in zip(bars, shift_rev.values):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'₹{val:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
        with col_b:
            if chan_col and rev_col:
                sh("📱","Dine In vs Delivery")
                chan_rev = df_rest.groupby(chan_col)[rev_col].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(5,4))
                clean_chart(fig, ax)
                ax.pie(chan_rev.values, labels=chan_rev.index, autopct='%1.1f%%',
                       colors=['#6366f1','#f97316','#10b981','#818cf8'],
                       wedgeprops=dict(edgecolor='white', linewidth=2), startangle=90)
                ax.set_title("Revenue by Channel", fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab4:
        if staff_col and rev_col:
            sh("👨‍🍳","Staff Performance")
            staff_stats = df_rest.groupby(staff_col).agg(
                Orders=(rev_col,'count'), Revenue=(rev_col,'sum')
            ).reset_index()
            if rating_col:
                staff_rating = df_rest.groupby(staff_col)[rating_col].mean().reset_index()
                staff_stats = staff_stats.merge(staff_rating, on=staff_col)
            staff_stats = staff_stats.sort_values('Revenue', ascending=False).reset_index(drop=True)
            col_a, col_b = st.columns(2)
            with col_a:
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                ax.bar(staff_stats[staff_col], staff_stats['Revenue'], color='#6366f1', width=0.5)
                ax.set_title("Revenue per Staff Member", fontweight='bold')
                plt.xticks(rotation=30, ha='right', fontsize=9)
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                disp = staff_stats.copy()
                disp['Revenue'] = disp['Revenue'].apply(lambda x: f"₹{x:,.0f}")
                st.dataframe(disp, use_container_width=True, hide_index=True)

    with tab5:
        rest_alerts = []
        if avg_food_cost > 45: rest_alerts.append(('red', f"🔴 Food cost % is <b>{avg_food_cost:.1f}%</b> — above 45%. Margins are being destroyed. Review pricing urgently."))
        elif avg_food_cost > 35: rest_alerts.append(('amber', f"⚠️ Food cost % is <b>{avg_food_cost:.1f}%</b> — above 35% target. Reduce portion sizes or increase prices."))
        if occupancy_rate < 50: rest_alerts.append(('amber', f"⚠️ Table occupancy is only <b>{occupancy_rate:.1f}%</b> — well below 70% target. Drive footfall."))
        if avg_rating < 3.5 and avg_rating > 0: rest_alerts.append(('red', f"🔴 Average rating is <b>{avg_rating:.1f}</b> — investigate food quality and service."))
        if total_waste > total_orders * 0.5: rest_alerts.append(('amber', f"⚠️ High wastage of <b>{total_waste:,.0f} units</b> — review ordering and portion control."))
        if not rest_alerts: abox("✅ Restaurant operations look healthy — no critical alerts.", "green")
        else:
            for k, m in rest_alerts: abox(m, k)
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# MODULE: HEALTHCARE INTELLIGENCE
# ════════════════════════════════════════════════════════════════════════════
if module == "Healthcare":
    df_hc = df.copy()
    show_clean_report(fixes, df_hc, df_raw)

    date_col   = next((c for c in df_hc.columns if 'date' in c.lower()), None)
    dept_col   = next((c for c in df_hc.columns if 'department' in c.lower() or 'dept' in c.lower()), None)
    doc_col    = next((c for c in df_hc.columns if 'doctor' in c.lower()), None)
    diag_col   = next((c for c in df_hc.columns if 'diagnosis' in c.lower()), None)
    ptype_col  = next((c for c in df_hc.columns if 'patient type' in c.lower()), None)
    pay_col    = next((c for c in df_hc.columns if 'payment' in c.lower()), None)
    rev_col    = next((c for c in df_hc.columns if 'revenue' in c.lower()), None)
    budget_col = next((c for c in df_hc.columns if 'budget' in c.lower()), None)
    bed_col    = next((c for c in df_hc.columns if 'bed days' in c.lower()), None)
    avail_col  = next((c for c in df_hc.columns if 'beds available' in c.lower()), None)
    readmit_col= next((c for c in df_hc.columns if 'readmit' in c.lower()), None)
    rating_col = next((c for c in df_hc.columns if 'rating' in c.lower()), None)
    city_col   = next((c for c in df_hc.columns if 'city' in c.lower()), None)

    if date_col: df_hc[date_col] = pd.to_datetime(df_hc[date_col], dayfirst=True, errors='coerce')

    total_rev     = nc(df_hc[rev_col]).sum() if rev_col else 0
    total_patients= len(df_hc)
    avg_rating    = nc(df_hc[rating_col]).mean() or 0 if rating_col else 0
    readmit_rate  = (df_hc[readmit_col].astype(str).str.upper()=='YES').mean()*100 if readmit_col else 0
    num_doctors   = df_hc[doc_col].nunique() if doc_col else 0
    budget_var    = (nc(df_hc[rev_col]).sum() - nc(df_hc[budget_col]).sum()) if rev_col and budget_col else 0

    kpi_row([
        (f"₹{total_rev:,.0f}", "Total Revenue", "indigo", None),
        (f"{total_patients:,}", "Total Patients", "indigo", None),
        (f"{num_doctors}", "Doctors", "indigo", None),
        (f"{readmit_rate:.1f}%", "Readmission Rate", "red" if readmit_rate>10 else "amber" if readmit_rate>5 else "green", "Target: <5%"),
        (f"{avg_rating:.1f}★", "Patient Rating", "green" if avg_rating>=4 else "amber", "Out of 5.0"),
        (f"₹{abs(budget_var):,.0f}", "Budget Variance", "green" if budget_var>=0 else "red", "Surplus" if budget_var>=0 else "Deficit"),
    ])

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏥 Department Analysis", "👨‍⚕️ Doctor Performance", "💊 Patient Insights",
        "💰 Revenue & Budget", "🚨 Smart Alerts"
    ])

    with tab1:
        sh("🏥","Revenue by Department")
        if dept_col and rev_col:
            col_a, col_b = st.columns(2)
            with col_a:
                dept_rev = df_hc.groupby(dept_col)[rev_col].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                ax.barh(dept_rev.index, dept_rev.values, color='#6366f1')
                ax.invert_yaxis()
                ax.set_title("Revenue by Department", fontweight='bold')
                for i,val in enumerate(dept_rev.values):
                    ax.text(val, i, f'  ₹{val:,.0f}', va='center', fontsize=9)
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                dept_count = df_hc[dept_col].value_counts()
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                ax.pie(dept_count.values, labels=dept_count.index, autopct='%1.1f%%',
                       colors=['#6366f1','#818cf8','#a5b4fc','#c7d2fe','#e0e7ff','#4f46e5','#10b981'],
                       wedgeprops=dict(edgecolor='white',linewidth=2))
                ax.set_title("Patient Volume by Department", fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()

        if dept_col and rev_col and budget_col:
            sh("📊","Department Budget vs Actual")
            dept_bva = df_hc.groupby(dept_col).agg({rev_col:'sum', budget_col:'sum'}).reset_index()
            dept_bva.columns = ['Department','Actual Revenue','Budget']
            dept_bva['Variance'] = dept_bva['Actual Revenue'] - dept_bva['Budget']
            fig, ax = plt.subplots(figsize=(10,4))
            clean_chart(fig, ax)
            x = range(len(dept_bva))
            ax.bar([i-0.2 for i in x], dept_bva['Budget'], 0.35, label='Budget', color='#e0e7ff', edgecolor='white')
            ax.bar([i+0.2 for i in x], dept_bva['Actual Revenue'], 0.35, label='Actual',
                   color=['#10b981' if v>=0 else '#ef4444' for v in dept_bva['Variance']], edgecolor='white')
            ax.set_xticks(list(x)); ax.set_xticklabels(dept_bva['Department'], rotation=20, ha='right')
            ax.legend(); ax.set_title("Budget vs Actual by Department", fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        if doc_col and rev_col:
            sh("👨‍⚕️","Doctor Performance")
            doc_stats = df_hc.groupby(doc_col).agg(
                Patients=(rev_col,'count'), Revenue=(rev_col,'sum')
            ).reset_index()
            if rating_col:
                doc_rating = df_hc.groupby(doc_col)[rating_col].mean().reset_index()
                doc_stats = doc_stats.merge(doc_rating, on=doc_col)
            doc_stats = doc_stats.sort_values('Revenue', ascending=False).reset_index(drop=True)
            col_a, col_b = st.columns(2)
            with col_a:
                fig, ax = plt.subplots(figsize=(6,5))
                clean_chart(fig, ax)
                ax.barh(doc_stats[doc_col], doc_stats['Revenue'], color='#6366f1')
                ax.invert_yaxis()
                ax.set_title("Revenue per Doctor", fontweight='bold')
                for i,val in enumerate(doc_stats['Revenue']):
                    ax.text(val, i, f'  ₹{val:,.0f}', va='center', fontsize=8)
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                disp = doc_stats.copy()
                disp['Revenue'] = disp['Revenue'].apply(lambda x: f"₹{x:,.0f}")
                st.dataframe(disp, use_container_width=True, hide_index=True)

    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            if diag_col:
                sh("🔬","Top Diagnoses")
                diag_count = df_hc[diag_col].value_counts().head(10)
                fig, ax = plt.subplots(figsize=(6,5))
                clean_chart(fig, ax)
                diag_count.plot(kind='barh', ax=ax, color='#818cf8')
                ax.invert_yaxis()
                ax.set_title("Most Common Diagnoses", fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
        with col_b:
            if ptype_col:
                sh("👥","Patient Type Breakdown")
                pt_count = df_hc[ptype_col].value_counts()
                fig, ax = plt.subplots(figsize=(5,5))
                clean_chart(fig, ax)
                ax.pie(pt_count.values, labels=pt_count.index, autopct='%1.1f%%',
                       colors=['#6366f1','#10b981','#ef4444'],
                       wedgeprops=dict(edgecolor='white',linewidth=2))
                ax.set_title("Patient Type Split", fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
        if pay_col and rev_col:
            sh("💳","Revenue by Payment Type")
            pay_rev = df_hc.groupby(pay_col)[rev_col].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10,3))
            clean_chart(fig, ax)
            bars = ax.bar(pay_rev.index, pay_rev.values, color=['#6366f1','#818cf8','#a5b4fc','#c7d2fe'], width=0.5)
            for bar, val in zip(bars, pay_rev.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'₹{val:,.0f}', ha='center', va='bottom', fontsize=9)
            ax.set_title("Revenue by Payment Type", fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab4:
        if date_col and rev_col:
            sh("📈","Monthly Revenue Trend")
            monthly = df_hc.groupby(df_hc[date_col].dt.to_period('M'))[rev_col].sum().reset_index()
            monthly.columns = ['Month','Revenue']; monthly['Month'] = monthly['Month'].astype(str)
            fig, ax = plt.subplots(figsize=(12,4))
            clean_chart(fig, ax)
            ax.fill_between(range(len(monthly)), monthly['Revenue'], alpha=0.12, color='#6366f1')
            ax.plot(range(len(monthly)), monthly['Revenue'], color='#6366f1', linewidth=2.5, marker='o', markersize=5)
            ax.set_xticks(range(len(monthly))); ax.set_xticklabels(monthly['Month'], rotation=45, ha='right', fontsize=8)
            ax.set_title("Monthly Revenue Trend", fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab5:
        hc_alerts = []
        if readmit_rate > 10: hc_alerts.append(('red', f"🔴 Readmission rate <b>{readmit_rate:.1f}%</b> is very high — review patient discharge protocols."))
        elif readmit_rate > 5: hc_alerts.append(('amber', f"⚠️ Readmission rate <b>{readmit_rate:.1f}%</b> exceeds 5% target — needs attention."))
        if avg_rating < 3.5 and avg_rating > 0: hc_alerts.append(('red', f"🔴 Patient rating <b>{avg_rating:.1f}/5</b> is low — investigate service quality."))
        if budget_var < 0: hc_alerts.append(('amber', f"⚠️ Revenue is <b>₹{abs(budget_var):,.0f} below budget</b> — review department performance."))
        if not hc_alerts: abox("✅ Healthcare operations look healthy — no critical alerts.", "green")
        else:
            for k,m in hc_alerts: abox(m, k)
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# MODULE: MANUFACTURING & OPERATIONS INTELLIGENCE
# ════════════════════════════════════════════════════════════════════════════
if module == "Manufacturing":
    df_mfg = df.copy()
    show_clean_report(fixes, df_mfg, df_raw)

    date_col    = next((c for c in df_mfg.columns if 'date' in c.lower()), None)
    machine_col = next((c for c in df_mfg.columns if 'machine' in c.lower()), None)
    prod_col    = next((c for c in df_mfg.columns if 'product' in c.lower()), None)
    shift_col   = next((c for c in df_mfg.columns if 'shift' in c.lower()), None)
    plan_col    = next((c for c in df_mfg.columns if 'planned' in c.lower()), None)
    actual_col  = next((c for c in df_mfg.columns if 'actual output' in c.lower()), None)
    defect_col  = next((c for c in df_mfg.columns if 'defect' in c.lower()), None)
    down_col    = next((c for c in df_mfg.columns if 'downtime' in c.lower() and 'hour' in c.lower()), None)
    dcause_col  = next((c for c in df_mfg.columns if 'cause' in c.lower()), None)
    dcost_col   = next((c for c in df_mfg.columns if 'downtime cost' in c.lower()), None)
    energy_col  = next((c for c in df_mfg.columns if 'energy' in c.lower()), None)
    labour_col  = next((c for c in df_mfg.columns if 'labour' in c.lower()), None)
    material_col= next((c for c in df_mfg.columns if 'material' in c.lower()), None)
    oper_col    = next((c for c in df_mfg.columns if 'operator' in c.lower()), None)

    if date_col: df_mfg[date_col] = pd.to_datetime(df_mfg[date_col], dayfirst=True, errors='coerce')

    total_planned = nc(df_mfg[plan_col]).sum() if plan_col else 0
    total_actual  = nc(df_mfg[actual_col]).sum() if actual_col else 0
    efficiency    = (total_actual/total_planned*100) if total_planned > 0 else 0
    total_defects = nc(df_mfg[defect_col]).sum() if defect_col else 0
    defect_rate   = (total_defects/total_actual*100) if total_actual > 0 else 0
    total_downtime= nc(df_mfg[down_col]).sum() if down_col else 0
    total_dt_cost = nc(df_mfg[dcost_col]).sum() if dcost_col else 0
    total_cost    = (nc(df_mfg[labour_col]).sum() if labour_col else 0) + (nc(df_mfg[material_col]).sum() if material_col else 0)

    kpi_row([
        (f"{efficiency:.1f}%", "Production Efficiency", "green" if efficiency>=90 else "amber" if efficiency>=75 else "red", "Target: 90%+"),
        (f"{total_actual:,.0f}", "Units Produced", "indigo", f"of {total_planned:,.0f} planned"),
        (f"{defect_rate:.1f}%", "Defect Rate", "green" if defect_rate<3 else "amber" if defect_rate<6 else "red", "Target: <3%"),
        (f"{total_downtime:.0f} hrs", "Total Downtime", "red" if total_downtime>50 else "amber" if total_downtime>20 else "green", "Review causes"),
        (f"₹{total_dt_cost:,.0f}", "Downtime Cost", "red" if total_dt_cost>100000 else "amber", "Lost production"),
        (f"₹{total_cost:,.0f}", "Total Prod. Cost", "indigo", "Labour + Material"),
    ])

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⚙️ Production Output", "🔧 Downtime Analysis", "🏭 Machine & Shift",
        "💰 Cost Analysis", "🚨 Smart Alerts"
    ])

    with tab1:
        sh("⚙️","Production Efficiency")
        if plan_col and actual_col:
            col_a, col_b = st.columns(2)
            with col_a:
                if machine_col:
                    mach_eff = df_mfg.groupby(machine_col).apply(
                        lambda x: (x[actual_col].sum()/x[plan_col].sum()*100) if x[plan_col].sum()>0 else 0
                    ).sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(6,4))
                    clean_chart(fig, ax)
                    bar_c = ['#10b981' if v>=90 else '#f59e0b' if v>=75 else '#ef4444' for v in mach_eff.values]
                    ax.barh(mach_eff.index, mach_eff.values, color=bar_c)
                    ax.axvline(90, color='#6366f1', linestyle='--', linewidth=1.5, label='Target 90%')
                    ax.set_title("Efficiency by Machine", fontweight='bold')
                    ax.set_xlabel("Efficiency %")
                    ax.legend(fontsize=9)
                    for i,val in enumerate(mach_eff.values):
                        ax.text(val+0.3, i, f'{val:.1f}%', va='center', fontsize=9)
                    plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                if shift_col:
                    shift_eff = df_mfg.groupby(shift_col).apply(
                        lambda x: (x[actual_col].sum()/x[plan_col].sum()*100) if x[plan_col].sum()>0 else 0
                    )
                    fig, ax = plt.subplots(figsize=(5,4))
                    clean_chart(fig, ax)
                    bar_c = ['#f59e0b','#6366f1','#1e1b4b']
                    ax.bar(shift_eff.index, shift_eff.values, color=bar_c[:len(shift_eff)], width=0.4)
                    ax.axhline(90, color='#ef4444', linestyle='--', linewidth=1.5, label='Target 90%')
                    ax.set_title("Efficiency by Shift", fontweight='bold')
                    ax.legend(fontsize=9)
                    for i,(idx,val) in enumerate(shift_eff.items()):
                        ax.text(i, val+0.5, f'{val:.1f}%', ha='center', fontweight='bold', fontsize=10)
                    plt.tight_layout(); st.pyplot(fig); plt.close()

        if date_col and actual_col:
            sh("📈","Daily Production Trend")
            daily = df_mfg.groupby(df_mfg[date_col].dt.to_period('M')).agg(
                {plan_col:'sum', actual_col:'sum'}).reset_index()
            daily.columns = ['Month','Planned','Actual']
            daily['Month'] = daily['Month'].astype(str)
            fig, ax = plt.subplots(figsize=(12,4))
            clean_chart(fig, ax)
            ax.plot(range(len(daily)), daily['Planned'], color='#e0e7ff', linewidth=2, linestyle='--', label='Planned', marker='s', markersize=4)
            ax.plot(range(len(daily)), daily['Actual'], color='#6366f1', linewidth=2.5, label='Actual', marker='o', markersize=5)
            ax.fill_between(range(len(daily)), daily['Planned'], daily['Actual'],
                           where=[a<p for a,p in zip(daily['Actual'],daily['Planned'])],
                           alpha=0.12, color='#ef4444', label='Shortfall')
            ax.set_xticks(range(len(daily))); ax.set_xticklabels(daily['Month'], rotation=45, ha='right', fontsize=8)
            ax.set_title("Planned vs Actual Production", fontweight='bold')
            ax.legend(fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        sh("🔧","Downtime Analysis")
        if down_col:
            col_a, col_b = st.columns(2)
            with col_a:
                if machine_col:
                    mach_down = df_mfg.groupby(machine_col)[down_col].sum().sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(6,4))
                    clean_chart(fig, ax)
                    ax.barh(mach_down.index, mach_down.values, color='#ef4444', alpha=0.85)
                    ax.invert_yaxis()
                    ax.set_title("Downtime Hours by Machine", fontweight='bold')
                    ax.set_xlabel("Hours")
                    for i,val in enumerate(mach_down.values):
                        ax.text(val+0.1, i, f'{val:.1f}h', va='center', fontsize=9)
                    plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                if dcause_col:
                    cause_down = df_mfg[df_mfg[dcause_col].astype(str).str.lower()!='none'].groupby(dcause_col)[down_col].sum().sort_values(ascending=False)
                    if len(cause_down) > 0:
                        fig, ax = plt.subplots(figsize=(5,5))
                        clean_chart(fig, ax)
                        ax.pie(cause_down.values, labels=cause_down.index, autopct='%1.1f%%',
                               colors=['#ef4444','#f59e0b','#6366f1','#818cf8'],
                               wedgeprops=dict(edgecolor='white',linewidth=2))
                        ax.set_title("Downtime by Cause", fontweight='bold')
                        plt.tight_layout(); st.pyplot(fig); plt.close()

        if defect_col and actual_col:
            sh("🔍","Quality & Defect Analysis")
            col_c, col_d = st.columns(2)
            with col_c:
                if machine_col:
                    mach_defect = df_mfg.groupby(machine_col).apply(
                        lambda x: (x[defect_col].sum()/x[actual_col].sum()*100) if x[actual_col].sum()>0 else 0
                    ).sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(6,4))
                    clean_chart(fig, ax)
                    bar_c = ['#ef4444' if v>6 else '#f59e0b' if v>3 else '#10b981' for v in mach_defect.values]
                    ax.barh(mach_defect.index, mach_defect.values, color=bar_c)
                    ax.axvline(3, color='#6366f1', linestyle='--', linewidth=1.5, label='Target 3%')
                    ax.set_title("Defect Rate by Machine", fontweight='bold')
                    ax.legend(fontsize=9)
                    plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_d:
                if shift_col:
                    shift_defect = df_mfg.groupby(shift_col).apply(
                        lambda x: (x[defect_col].sum()/x[actual_col].sum()*100) if x[actual_col].sum()>0 else 0
                    )
                    fig, ax = plt.subplots(figsize=(5,4))
                    clean_chart(fig, ax)
                    bar_c = ['#f59e0b','#6366f1','#1e1b4b']
                    ax.bar(shift_defect.index, shift_defect.values, color=bar_c[:len(shift_defect)], width=0.4)
                    ax.axhline(3, color='#ef4444', linestyle='--', linewidth=1.5, label='Target 3%')
                    ax.set_title("Defect Rate by Shift", fontweight='bold')
                    ax.legend(fontsize=9)
                    plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        if machine_col and actual_col and plan_col:
            sh("🏭","Machine Summary")
            mach_summary = df_mfg.groupby(machine_col).agg({
                plan_col:'sum', actual_col:'sum'
            }).reset_index()
            mach_summary.columns = ['Machine','Planned','Actual']
            mach_summary['Efficiency %'] = (mach_summary['Actual']/mach_summary['Planned']*100).round(1)
            if down_col:
                mach_down2 = df_mfg.groupby(machine_col)[down_col].sum().reset_index()
                mach_down2.columns = ['Machine','Downtime (hrs)']
                mach_summary = mach_summary.merge(mach_down2, on='Machine')
            if defect_col:
                mach_def2 = df_mfg.groupby(machine_col)[defect_col].sum().reset_index()
                mach_def2.columns = ['Machine','Total Defects']
                mach_summary = mach_summary.merge(mach_def2, on='Machine')
            st.dataframe(
                mach_summary.style.background_gradient(subset=['Efficiency %'], cmap='RdYlGn'),
                use_container_width=True, hide_index=True
            )

    with tab4:
        sh("💰","Cost Breakdown")
        cost_items = {}
        if labour_col: cost_items['Labour'] = nc(df_mfg[labour_col]).sum()
        if material_col: cost_items['Material'] = nc(df_mfg[material_col]).sum()
        if dcost_col: cost_items['Downtime Loss'] = nc(df_mfg[dcost_col]).sum()
        if energy_col: cost_items['Energy'] = nc(df_mfg[energy_col]).sum() * 8  # assume ₹8/kwh

        if cost_items:
            col_a, col_b = st.columns(2)
            with col_a:
                fig, ax = plt.subplots(figsize=(5,5))
                clean_chart(fig, ax)
                ax.pie(list(cost_items.values()), labels=list(cost_items.keys()), autopct='%1.1f%%',
                       colors=['#6366f1','#818cf8','#ef4444','#f59e0b'],
                       wedgeprops=dict(edgecolor='white',linewidth=2))
                ax.set_title("Cost Breakdown", fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                for k,v in cost_items.items():
                    pct = v/sum(cost_items.values())*100
                    st.markdown(f"<div style='display:flex;justify-content:space-between;background:white;border-radius:10px;padding:12px 16px;margin:6px 0;border-left:4px solid #6366f1;box-shadow:0 1px 4px rgba(0,0,0,0.05);'><span style='font-weight:600'>{k}</span><span style='font-weight:800;color:#6366f1'>₹{v:,.0f} <span style='font-size:0.75rem;color:#6b7280'>({pct:.1f}%)</span></span></div>", unsafe_allow_html=True)

    with tab5:
        mfg_alerts = []
        if efficiency < 75: mfg_alerts.append(('red', f"🔴 Production efficiency <b>{efficiency:.1f}%</b> is critically low — review machine performance and scheduling."))
        elif efficiency < 90: mfg_alerts.append(('amber', f"⚠️ Production efficiency <b>{efficiency:.1f}%</b> below 90% target."))
        if defect_rate > 6: mfg_alerts.append(('red', f"🔴 Defect rate <b>{defect_rate:.1f}%</b> is very high — immediate quality review needed."))
        elif defect_rate > 3: mfg_alerts.append(('amber', f"⚠️ Defect rate <b>{defect_rate:.1f}%</b> exceeds 3% target."))
        if total_downtime > 50: mfg_alerts.append(('amber', f"⚠️ <b>{total_downtime:.0f} hours</b> of downtime — ₹{total_dt_cost:,.0f} lost. Review maintenance schedule."))
        if not mfg_alerts: abox("✅ Manufacturing operations look healthy — no critical alerts.", "green")
        else:
            for k,m in mfg_alerts: abox(m, k)
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# MODULE: MARKETING INTELLIGENCE
# ════════════════════════════════════════════════════════════════════════════
if module == "Marketing":
    df_mkt = df.copy()
    show_clean_report(fixes, df_mkt, df_raw)

    date_col   = next((c for c in df_mkt.columns if 'date' in c.lower()), None)
    chan_col   = next((c for c in df_mkt.columns if 'channel' in c.lower()), None)
    camp_col   = next((c for c in df_mkt.columns if 'campaign' in c.lower()), None)
    spend_col  = next((c for c in df_mkt.columns if 'spend' in c.lower()), None)
    imp_col    = next((c for c in df_mkt.columns if 'impression' in c.lower()), None)
    click_col  = next((c for c in df_mkt.columns if 'click' in c.lower() and 'rate' not in c.lower()), None)
    lead_col   = next((c for c in df_mkt.columns if 'lead' in c.lower()), None)
    conv_col   = next((c for c in df_mkt.columns if 'conversion' in c.lower()), None)
    rev_col    = next((c for c in df_mkt.columns if 'revenue' in c.lower()), None)
    cac_col    = next((c for c in df_mkt.columns if 'cac' in c.lower()), None)
    roas_col   = next((c for c in df_mkt.columns if 'roas' in c.lower()), None)
    city_col   = next((c for c in df_mkt.columns if 'city' in c.lower()), None)

    if date_col: df_mkt[date_col] = pd.to_datetime(df_mkt[date_col], dayfirst=True, errors='coerce')

    total_spend = nc(df_mkt[spend_col]).sum() if spend_col else 0
    total_rev   = nc(df_mkt[rev_col]).sum() if rev_col else 0
    total_leads = nc(df_mkt[lead_col]).sum() if lead_col else 0
    total_conv  = nc(df_mkt[conv_col]).sum() if conv_col else 0
    overall_roas= round(total_rev/total_spend, 2) if total_spend > 0 else 0
    avg_cac     = round(total_spend/total_conv, 0) if total_conv > 0 else 0
    ctr         = round(nc(df_mkt[click_col]).sum()/nc(df_mkt[imp_col]).sum()*100, 2) if click_col and imp_col else 0
    conv_rate   = round(total_conv/total_leads*100, 1) if total_leads > 0 else 0

    kpi_row([
        (f"₹{total_spend:,.0f}", "Total Ad Spend", "indigo", None),
        (f"₹{total_rev:,.0f}", "Revenue Generated", "green", None),
        (f"{overall_roas}x", "Overall ROAS", "green" if overall_roas>=3 else "amber" if overall_roas>=2 else "red", "Target: 3x+"),
        (f"₹{avg_cac:,.0f}", "Avg CAC", "green" if avg_cac<5000 else "amber", "Cost per customer"),
        (f"{total_leads:,.0f}", "Total Leads", "indigo", None),
        (f"{conv_rate:.1f}%", "Lead→Sale Rate", "green" if conv_rate>=20 else "amber" if conv_rate>=10 else "red", "Conversion rate"),
    ])

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Channel Performance", "📣 Campaign Analysis", "🔄 Funnel & Conversion",
        "📈 Trends", "🚨 Smart Alerts"
    ])

    with tab1:
        sh("📊","Channel Performance")
        if chan_col and spend_col:
            col_a, col_b = st.columns(2)
            with col_a:
                chan_spend = df_mkt.groupby(chan_col)[spend_col].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                ax.pie(chan_spend.values, labels=chan_spend.index, autopct='%1.1f%%',
                       colors=['#6366f1','#818cf8','#a5b4fc','#c7d2fe','#e0e7ff','#4f46e5','#10b981','#34d399'],
                       wedgeprops=dict(edgecolor='white',linewidth=2))
                ax.set_title("Budget Split by Channel", fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                if rev_col and conv_col:
                    chan_stats = df_mkt.groupby(chan_col).agg({spend_col:'sum', rev_col:'sum', conv_col:'sum'}).reset_index()
                    chan_stats.columns = ['Channel','Spend','Revenue','Conversions']
                    chan_stats['ROAS'] = (chan_stats['Revenue']/chan_stats['Spend']).round(2)
                    chan_stats['CAC'] = (chan_stats['Spend']/chan_stats['Conversions'].replace(0,1)).round(0)
                    chan_stats = chan_stats.sort_values('ROAS', ascending=False)
                    st.dataframe(chan_stats.style.background_gradient(subset=['ROAS'], cmap='Greens'), use_container_width=True, hide_index=True)
                else:
                    chan_stats = df_mkt.groupby(chan_col)[spend_col].sum().reset_index()
                    st.dataframe(chan_stats, use_container_width=True, hide_index=True)

            if rev_col and chan_col:
                sh("💰","Revenue by Channel")
                chan_rev = df_mkt.groupby(chan_col)[rev_col].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10,3))
                clean_chart(fig, ax)
                bars = ax.bar(chan_rev.index, chan_rev.values, color='#6366f1', alpha=0.9, width=0.5)
                for bar, val in zip(bars, chan_rev.values):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'₹{val:,.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
                ax.set_title("Revenue Generated by Channel", fontweight='bold')
                plt.xticks(rotation=20, ha='right')
                plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        if camp_col and spend_col:
            sh("📣","Campaign Performance")
            camp_stats = df_mkt.groupby(camp_col).agg({spend_col:'sum'}).reset_index()
            camp_stats.columns = ['Campaign','Total Spend']
            if rev_col:
                camp_rev = df_mkt.groupby(camp_col)[rev_col].sum().reset_index()
                camp_rev.columns = ['Campaign','Revenue']
                camp_stats = camp_stats.merge(camp_rev, on='Campaign')
                camp_stats['ROAS'] = (camp_stats['Revenue']/camp_stats['Total Spend']).round(2)
            if conv_col:
                camp_conv = df_mkt.groupby(camp_col)[conv_col].sum().reset_index()
                camp_conv.columns = ['Campaign','Conversions']
                camp_stats = camp_stats.merge(camp_conv, on='Campaign')
            camp_stats = camp_stats.sort_values('ROAS' if 'ROAS' in camp_stats.columns else 'Total Spend', ascending=False)

            col_a, col_b = st.columns(2)
            with col_a:
                if 'ROAS' in camp_stats.columns:
                    fig, ax = plt.subplots(figsize=(6,5))
                    clean_chart(fig, ax)
                    bar_c = ['#10b981' if v>=3 else '#f59e0b' if v>=2 else '#ef4444' for v in camp_stats['ROAS']]
                    ax.barh(camp_stats['Campaign'], camp_stats['ROAS'], color=bar_c)
                    ax.axvline(3, color='#6366f1', linestyle='--', linewidth=1.5, label='Target 3x')
                    ax.set_title("ROAS by Campaign", fontweight='bold')
                    ax.legend(fontsize=9)
                    plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                st.dataframe(camp_stats.reset_index(drop=True), use_container_width=True, hide_index=True)

    with tab3:
        sh("🔄","Marketing Funnel")
        if imp_col and click_col and lead_col and conv_col:
            funnel_data = {
                'Impressions': nc(df_mkt[imp_col]).sum(),
                'Clicks': nc(df_mkt[click_col]).sum(),
                'Leads': nc(df_mkt[lead_col]).sum(),
                'Conversions': nc(df_mkt[conv_col]).sum(),
            }
            fig, ax = plt.subplots(figsize=(8,5))
            clean_chart(fig, ax)
            colors_f = ['#6366f1','#818cf8','#a5b4fc','#10b981']
            bars = ax.barh(list(funnel_data.keys()), list(funnel_data.values()), color=colors_f, height=0.5)
            for bar, (label, val) in zip(bars, funnel_data.items()):
                ax.text(val+max(funnel_data.values())*0.01, bar.get_y()+bar.get_height()/2,
                        f'{val:,.0f}', va='center', fontweight='bold', fontsize=11)
            ax.set_title("Marketing Funnel — Impressions to Conversions", fontweight='bold')
            ax.invert_yaxis()
            plt.tight_layout(); st.pyplot(fig); plt.close()

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Click-Through Rate", f"{ctr:.2f}%", "Impressions → Clicks")
            col_b.metric("Lead Conversion", f"{conv_rate:.1f}%", "Leads → Sales")
            col_c.metric("Cost per Lead", f"₹{total_spend/total_leads:.0f}" if total_leads > 0 else "N/A", "Total spend / leads")

    with tab4:
        if date_col and spend_col and rev_col:
            sh("📈","Spend vs Revenue Trend")
            monthly = df_mkt.groupby(df_mkt[date_col].dt.to_period('M')).agg({spend_col:'sum', rev_col:'sum'}).reset_index()
            monthly.columns = ['Month','Spend','Revenue']
            monthly['Month'] = monthly['Month'].astype(str)
            fig, ax = plt.subplots(figsize=(12,4))
            clean_chart(fig, ax)
            ax2 = ax.twinx()
            ax.bar(range(len(monthly)), monthly['Spend'], color='#e0e7ff', label='Spend', alpha=0.8)
            ax2.plot(range(len(monthly)), monthly['Revenue'], color='#6366f1', linewidth=2.5, marker='o', markersize=5, label='Revenue')
            ax.set_xticks(range(len(monthly))); ax.set_xticklabels(monthly['Month'], rotation=45, ha='right', fontsize=8)
            ax.set_ylabel("Spend (₹)", color='#9ca3af'); ax2.set_ylabel("Revenue (₹)", color='#6366f1')
            ax.set_title("Monthly Spend vs Revenue", fontweight='bold')
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1+lines2, labels1+labels2, fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab5:
        mkt_alerts = []
        if overall_roas < 2: mkt_alerts.append(('red', f"🔴 Overall ROAS is <b>{overall_roas}x</b> — below 2x minimum. Ad spend is not generating enough revenue."))
        elif overall_roas < 3: mkt_alerts.append(('amber', f"⚠️ ROAS <b>{overall_roas}x</b> is below 3x target. Optimise underperforming campaigns."))
        if avg_cac > 10000: mkt_alerts.append(('amber', f"⚠️ Customer acquisition cost <b>₹{avg_cac:,.0f}</b> is high. Review channel efficiency."))
        if conv_rate < 10: mkt_alerts.append(('amber', f"⚠️ Lead-to-sale rate only <b>{conv_rate:.1f}%</b> — sales follow-up process needs improvement."))
        if not mkt_alerts: abox("✅ Marketing performance looks healthy — no critical alerts.", "green")
        else:
            for k,m in mkt_alerts: abox(m, k)
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# MODULE: EDUCATION INTELLIGENCE
# ════════════════════════════════════════════════════════════════════════════
if module == "Education":
    df_edu = df.copy()
    show_clean_report(fixes, df_edu, df_raw)

    date_col    = next((c for c in df_edu.columns if 'date' in c.lower()), None)
    class_col   = next((c for c in df_edu.columns if 'class' in c.lower()), None)
    subj_col    = next((c for c in df_edu.columns if 'subject' in c.lower()), None)
    teacher_col = next((c for c in df_edu.columns if 'teacher' in c.lower()), None)
    total_s_col = next((c for c in df_edu.columns if 'total student' in c.lower()), None)
    present_col = next((c for c in df_edu.columns if 'present' in c.lower()), None)
    score_col   = next((c for c in df_edu.columns if 'avg score' in c.lower() or 'score' in c.lower()), None)
    pass_col    = next((c for c in df_edu.columns if 'pass' in c.lower()), None)
    fee_col     = next((c for c in df_edu.columns if 'fee charged' in c.lower()), None)
    paid_col    = next((c for c in df_edu.columns if 'fee paid' in c.lower()), None)
    gender_col  = next((c for c in df_edu.columns if 'gender' in c.lower()), None)
    status_col  = next((c for c in df_edu.columns if 'status' in c.lower()), None)

    if date_col: df_edu[date_col] = pd.to_datetime(df_edu[date_col], dayfirst=True, errors='coerce')
    if total_s_col and present_col:
        df_edu['Attendance %'] = (df_edu[present_col]/df_edu[total_s_col]*100).round(1)

    total_records = len(df_edu)
    avg_score     = nc(df_edu[score_col]).mean() or 0 if score_col else 0
    avg_attend    = df_edu['Attendance %'].mean() if 'Attendance %' in df_edu.columns else 0
    pass_rate     = (df_edu[pass_col].astype(str).str.upper()=='YES').mean()*100 if pass_col else 0
    total_fee     = nc(df_edu[fee_col]).sum() if fee_col else 0
    total_paid    = nc(df_edu[paid_col]).sum() if paid_col else 0
    fee_collection= round(total_paid/total_fee*100, 1) if total_fee > 0 else 0
    dropout_rate  = (df_edu[status_col].astype(str).str.title()=='Dropped Out').mean()*100 if status_col else 0

    kpi_row([
        (f"{avg_score:.1f}%", "Avg Student Score", "green" if avg_score>=60 else "amber" if avg_score>=40 else "red", "Class average"),
        (f"{avg_attend:.1f}%", "Avg Attendance", "green" if avg_attend>=80 else "amber" if avg_attend>=70 else "red", "Target: 80%+"),
        (f"{pass_rate:.1f}%", "Pass Rate", "green" if pass_rate>=85 else "amber" if pass_rate>=70 else "red", "Target: 85%+"),
        (f"{fee_collection:.1f}%", "Fee Collection", "green" if fee_collection>=90 else "amber" if fee_collection>=75 else "red", f"₹{total_paid:,.0f} collected"),
        (f"{dropout_rate:.1f}%", "Dropout Rate", "red" if dropout_rate>10 else "amber" if dropout_rate>5 else "green", "Target: <5%"),
        (f"₹{total_fee-total_paid:,.0f}", "Fee Outstanding", "red" if (total_fee-total_paid)>0 else "green", "Needs collection"),
    ])

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📚 Academic Performance", "📅 Attendance Analysis", "👨‍🏫 Teacher Performance",
        "💰 Fee Collection", "🚨 Smart Alerts"
    ])

    with tab1:
        sh("📚","Score Analysis")
        col_a, col_b = st.columns(2)
        with col_a:
            if score_col:
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                ax.hist(df_edu[score_col].dropna(), bins=15, color='#6366f1', edgecolor='white', linewidth=1.5)
                ax.axvline(avg_score, color='#ef4444', linestyle='--', linewidth=2, label=f'Avg: {avg_score:.1f}%')
                ax.axvline(40, color='#f59e0b', linestyle=':', linewidth=1.5, label='Pass mark: 40%')
                ax.set_title("Score Distribution", fontweight='bold')
                ax.set_xlabel("Score %")
                ax.legend(fontsize=9)
                plt.tight_layout(); st.pyplot(fig); plt.close()
        with col_b:
            if subj_col and score_col:
                subj_score = df_edu.groupby(subj_col)[score_col].mean().sort_values()
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                bar_c = ['#ef4444' if v<50 else '#f59e0b' if v<65 else '#10b981' for v in subj_score.values]
                ax.barh(subj_score.index, subj_score.values, color=bar_c)
                ax.axvline(40, color='#ef4444', linestyle='--', linewidth=1.5, label='Pass: 40%')
                ax.set_title("Avg Score by Subject", fontweight='bold')
                ax.legend(fontsize=9)
                for i,val in enumerate(subj_score.values):
                    ax.text(val+0.3, i, f'{val:.1f}%', va='center', fontsize=9)
                plt.tight_layout(); st.pyplot(fig); plt.close()

        if class_col and score_col:
            sh("🏫","Performance by Class")
            class_perf = df_edu.groupby(class_col)[score_col].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10,3))
            clean_chart(fig, ax)
            bar_c = ['#10b981' if v>=65 else '#f59e0b' if v>=50 else '#ef4444' for v in class_perf.values]
            bars = ax.bar(class_perf.index, class_perf.values, color=bar_c, width=0.5)
            ax.axhline(40, color='#ef4444', linestyle='--', linewidth=1.5, label='Pass mark')
            for bar, val in zip(bars, class_perf.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
            ax.set_title("Average Score by Class", fontweight='bold')
            ax.legend(fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        sh("📅","Attendance Analysis")
        if 'Attendance %' in df_edu.columns:
            col_a, col_b = st.columns(2)
            with col_a:
                if class_col:
                    class_att = df_edu.groupby(class_col)['Attendance %'].mean().sort_values()
                    fig, ax = plt.subplots(figsize=(6,4))
                    clean_chart(fig, ax)
                    bar_c = ['#ef4444' if v<70 else '#f59e0b' if v<80 else '#10b981' for v in class_att.values]
                    ax.barh(class_att.index, class_att.values, color=bar_c)
                    ax.axvline(80, color='#6366f1', linestyle='--', linewidth=1.5, label='Target 80%')
                    ax.set_title("Attendance % by Class", fontweight='bold')
                    ax.legend(fontsize=9)
                    for i,val in enumerate(class_att.values):
                        ax.text(val+0.2, i, f'{val:.1f}%', va='center', fontsize=9)
                    plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                low_att = df_edu[df_edu['Attendance %'] < 70]
                if len(low_att) > 0:
                    abox(f"⚠️ <b>{len(low_att)} records</b> show attendance below 70% — risk of student disengagement.", "amber")
                high_att = (df_edu['Attendance %'] >= 90).sum()
                abox(f"✅ <b>{high_att} records</b> show 90%+ attendance — excellent engagement.", "green")

                fig, ax = plt.subplots(figsize=(5,4))
                clean_chart(fig, ax)
                ax.hist(df_edu['Attendance %'].dropna(), bins=15, color='#818cf8', edgecolor='white', linewidth=1.5)
                ax.axvline(80, color='#ef4444', linestyle='--', linewidth=1.5, label='Target 80%')
                ax.set_title("Attendance Distribution", fontweight='bold')
                ax.legend(fontsize=9)
                plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        if teacher_col and score_col:
            sh("👨‍🏫","Teacher Performance")
            teach_stats = df_edu.groupby(teacher_col).agg({
                score_col:'mean'
            }).reset_index()
            teach_stats.columns = ['Teacher','Avg Student Score']
            if 'Attendance %' in df_edu.columns:
                teach_att = df_edu.groupby(teacher_col)['Attendance %'].mean().reset_index()
                teach_att.columns = ['Teacher','Avg Attendance %']
                teach_stats = teach_stats.merge(teach_att, on='Teacher')
            teach_stats = teach_stats.sort_values('Avg Student Score', ascending=False).reset_index(drop=True)
            col_a, col_b = st.columns(2)
            with col_a:
                fig, ax = plt.subplots(figsize=(6,5))
                clean_chart(fig, ax)
                bar_c = ['#10b981' if v>=65 else '#f59e0b' if v>=50 else '#ef4444' for v in teach_stats['Avg Student Score']]
                ax.barh(teach_stats['Teacher'], teach_stats['Avg Student Score'], color=bar_c)
                ax.invert_yaxis()
                ax.set_title("Avg Student Score by Teacher", fontweight='bold')
                for i,val in enumerate(teach_stats['Avg Student Score']):
                    ax.text(val+0.2, i, f'{val:.1f}%', va='center', fontsize=9)
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                st.dataframe(teach_stats.style.background_gradient(subset=['Avg Student Score'], cmap='RdYlGn'), use_container_width=True, hide_index=True)

    with tab4:
        if fee_col and paid_col:
            sh("💰","Fee Collection Analysis")
            outstanding = total_fee - total_paid
            kpi_row([
                (f"₹{total_fee:,.0f}", "Total Fee Charged", "indigo", None),
                (f"₹{total_paid:,.0f}", "Fee Collected", "green", f"{fee_collection:.1f}%"),
                (f"₹{outstanding:,.0f}", "Outstanding", "red" if outstanding > 0 else "green", "Needs follow-up"),
            ])
            if class_col:
                sh("📋","Fee Collection by Class")
                class_fee = df_edu.groupby(class_col).agg({fee_col:'sum', paid_col:'sum'}).reset_index()
                class_fee.columns = ['Class','Total Fee','Collected']
                class_fee['Collection %'] = (class_fee['Collected']/class_fee['Total Fee']*100).round(1)
                class_fee['Outstanding'] = class_fee['Total Fee'] - class_fee['Collected']
                fig, ax = plt.subplots(figsize=(10,3))
                clean_chart(fig, ax)
                x = range(len(class_fee))
                ax.bar([i-0.2 for i in x], class_fee['Total Fee'], 0.35, label='Charged', color='#e0e7ff', edgecolor='white')
                ax.bar([i+0.2 for i in x], class_fee['Collected'], 0.35, label='Collected', color='#10b981', edgecolor='white')
                ax.set_xticks(list(x)); ax.set_xticklabels(class_fee['Class'], rotation=20, ha='right')
                ax.legend(); ax.set_title("Fee Charged vs Collected by Class", fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
                st.dataframe(class_fee.style.background_gradient(subset=['Collection %'], cmap='RdYlGn'), use_container_width=True, hide_index=True)

    with tab5:
        edu_alerts = []
        if avg_score < 50: edu_alerts.append(('red', f"🔴 Average score <b>{avg_score:.1f}%</b> is critically low — urgent academic intervention needed."))
        elif avg_score < 60: edu_alerts.append(('amber', f"⚠️ Average score <b>{avg_score:.1f}%</b> is below 60% — review teaching methods."))
        if avg_attend < 70: edu_alerts.append(('red', f"🔴 Average attendance <b>{avg_attend:.1f}%</b> is critically low."))
        elif avg_attend < 80: edu_alerts.append(('amber', f"⚠️ Attendance <b>{avg_attend:.1f}%</b> below 80% target."))
        if dropout_rate > 10: edu_alerts.append(('red', f"🔴 Dropout rate <b>{dropout_rate:.1f}%</b> is very high — investigate causes."))
        if fee_collection < 80: edu_alerts.append(('amber', f"⚠️ Fee collection rate <b>{fee_collection:.1f}%</b> — ₹{(total_fee-total_paid):,.0f} outstanding."))
        if not edu_alerts: abox("✅ School performance looks healthy — no critical alerts.", "green")
        else:
            for k,m in edu_alerts: abox(m, k)
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# MODULE: HOTEL & HOSPITALITY INTELLIGENCE
# ════════════════════════════════════════════════════════════════════════════
if module == "Hospitality":
    df_hot = df.copy()
    show_clean_report(fixes, df_hot, df_raw)

    date_col   = next((c for c in df_hot.columns if 'date' in c.lower()), None)
    rtype_col  = next((c for c in df_hot.columns if 'room type' in c.lower()), None)
    source_col = next((c for c in df_hot.columns if 'booking' in c.lower() or 'source' in c.lower()), None)
    city_col   = next((c for c in df_hot.columns if 'city' in c.lower()), None)
    nights_col = next((c for c in df_hot.columns if 'night' in c.lower()), None)
    rate_col   = next((c for c in df_hot.columns if 'room rate' in c.lower()), None)
    occ_col    = next((c for c in df_hot.columns if 'rooms occupied' in c.lower()), None)
    total_r_col= next((c for c in df_hot.columns if 'total rooms' in c.lower()), None)
    fnb_col    = next((c for c in df_hot.columns if 'f&b' in c.lower() or 'food' in c.lower()), None)
    spa_col    = next((c for c in df_hot.columns if 'spa' in c.lower()), None)
    rev_col    = next((c for c in df_hot.columns if 'total revenue' in c.lower()), None)
    rating_col = next((c for c in df_hot.columns if 'rating' in c.lower()), None)
    cancel_col = next((c for c in df_hot.columns if 'cancel' in c.lower()), None)
    staff_col  = next((c for c in df_hot.columns if 'staff cost' in c.lower()), None)
    opcost_col = next((c for c in df_hot.columns if 'operating' in c.lower()), None)

    if date_col: df_hot[date_col] = pd.to_datetime(df_hot[date_col], dayfirst=True, errors='coerce')
    if occ_col and total_r_col:
        df_hot['Occupancy %'] = (df_hot[occ_col]/df_hot[total_r_col]*100).round(1)

    total_rev     = nc(df_hot[rev_col]).sum() if rev_col else 0
    avg_occ       = df_hot['Occupancy %'].mean() if 'Occupancy %' in df_hot.columns else 0
    avg_rating    = nc(df_hot[rating_col]).mean() or 0 if rating_col else 0
    cancel_rate   = (df_hot[cancel_col].astype(str).str.upper()=='YES').mean()*100 if cancel_col else 0
    avg_rate      = nc(df_hot[rate_col]).mean() or 0 if rate_col else 0
    total_cost    = (nc(df_hot[staff_col]).sum() if staff_col else 0) + (nc(df_hot[opcost_col]).sum() if opcost_col else 0)
    net_profit    = total_rev - total_cost
    revpar        = round(avg_rate * avg_occ/100, 0)  # Revenue per available room

    kpi_row([
        (f"₹{total_rev:,.0f}", "Total Revenue", "indigo", None),
        (f"{avg_occ:.1f}%", "Avg Occupancy", "green" if avg_occ>=70 else "amber" if avg_occ>=50 else "red", "Target: 70%+"),
        (f"₹{avg_rate:,.0f}", "Avg Room Rate", "indigo", "ADR"),
        (f"₹{revpar:,.0f}", "RevPAR", "indigo", "Revenue per avail. room"),
        (f"{avg_rating:.1f}★", "Guest Rating", "green" if avg_rating>=4 else "amber", "Out of 5.0"),
        (f"{cancel_rate:.1f}%", "Cancellation Rate", "red" if cancel_rate>15 else "amber" if cancel_rate>8 else "green", "Target: <8%"),
    ])

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏨 Occupancy & Revenue","🛏️ Room Analysis","📱 Booking Sources","💰 Profitability","🚨 Smart Alerts"
    ])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            if date_col and rev_col:
                sh("📈","Monthly Revenue Trend")
                monthly = df_hot.groupby(df_hot[date_col].dt.to_period('M'))[rev_col].sum().reset_index()
                monthly.columns = ['Month','Revenue']; monthly['Month'] = monthly['Month'].astype(str)
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                ax.fill_between(range(len(monthly)), monthly['Revenue'], alpha=0.12, color='#6366f1')
                ax.plot(range(len(monthly)), monthly['Revenue'], color='#6366f1', linewidth=2.5, marker='o', markersize=5)
                ax.set_xticks(range(len(monthly))); ax.set_xticklabels(monthly['Month'], rotation=45, ha='right', fontsize=8)
                ax.set_title("Monthly Revenue", fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
        with col_b:
            if 'Occupancy %' in df_hot.columns and date_col:
                sh("📊","Monthly Occupancy Rate")
                monthly_occ = df_hot.groupby(df_hot[date_col].dt.to_period('M'))['Occupancy %'].mean().reset_index()
                monthly_occ.columns = ['Month','Occ %']; monthly_occ['Month'] = monthly_occ['Month'].astype(str)
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                bar_c = ['#10b981' if v>=70 else '#f59e0b' if v>=50 else '#ef4444' for v in monthly_occ['Occ %']]
                ax.bar(range(len(monthly_occ)), monthly_occ['Occ %'], color=bar_c, width=0.6)
                ax.axhline(70, color='#6366f1', linestyle='--', linewidth=1.5, label='Target 70%')
                ax.set_xticks(range(len(monthly_occ))); ax.set_xticklabels(monthly_occ['Month'], rotation=45, ha='right', fontsize=8)
                ax.legend(fontsize=9); ax.set_title("Monthly Occupancy %", fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()

        if fnb_col and spa_col and rev_col:
            sh("🍽️","Revenue Breakdown")
            room_rev = nc(df_hot[rate_col]).sum() if rate_col else 0
            fnb_rev  = nc(df_hot[fnb_col]).sum()
            spa_rev  = nc(df_hot[spa_col]).sum()
            other_rev = total_rev - room_rev - fnb_rev - spa_rev
            rev_breakdown = {'Room Revenue': room_rev, 'F&B': fnb_rev, 'Spa': spa_rev}
            if other_rev > 0: rev_breakdown['Other'] = other_rev
            fig, ax = plt.subplots(figsize=(6,4))
            clean_chart(fig, ax)
            ax.pie(list(rev_breakdown.values()), labels=list(rev_breakdown.keys()), autopct='%1.1f%%',
                   colors=['#6366f1','#f59e0b','#10b981','#818cf8'],
                   wedgeprops=dict(edgecolor='white',linewidth=2))
            ax.set_title("Revenue Source Breakdown", fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        if rtype_col and rev_col:
            sh("🛏️","Room Type Performance")
            col_a, col_b = st.columns(2)
            with col_a:
                rtype_rev = df_hot.groupby(rtype_col)[rev_col].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                ax.barh(rtype_rev.index, rtype_rev.values, color='#6366f1')
                ax.invert_yaxis()
                ax.set_title("Revenue by Room Type", fontweight='bold')
                for i,val in enumerate(rtype_rev.values):
                    ax.text(val,i,f'  ₹{val:,.0f}',va='center',fontsize=9)
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                rtype_stats = df_hot.groupby(rtype_col).agg({rev_col:'sum', rate_col:'mean'}).reset_index()
                rtype_stats.columns = ['Room Type','Total Revenue','Avg Rate']
                if rating_col:
                    rtype_rat = df_hot.groupby(rtype_col)[rating_col].mean().reset_index()
                    rtype_rat.columns = ['Room Type','Avg Rating']
                    rtype_stats = rtype_stats.merge(rtype_rat, on='Room Type')
                rtype_stats = rtype_stats.sort_values('Total Revenue', ascending=False)
                st.dataframe(rtype_stats.reset_index(drop=True), use_container_width=True, hide_index=True)

    with tab3:
        if source_col and rev_col:
            sh("📱","Booking Source Analysis")
            col_a, col_b = st.columns(2)
            with col_a:
                src_rev = df_hot.groupby(source_col)[rev_col].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(6,5))
                clean_chart(fig, ax)
                ax.pie(src_rev.values, labels=src_rev.index, autopct='%1.1f%%',
                       colors=['#6366f1','#818cf8','#a5b4fc','#c7d2fe','#e0e7ff','#4f46e5','#10b981'],
                       wedgeprops=dict(edgecolor='white',linewidth=2))
                ax.set_title("Revenue by Booking Source", fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                src_stats = df_hot.groupby(source_col).agg(Bookings=(rev_col,'count'), Revenue=(rev_col,'sum')).reset_index()
                src_stats['Avg Booking Value'] = (src_stats['Revenue']/src_stats['Bookings']).round(0)
                src_stats['Revenue'] = src_stats['Revenue'].apply(lambda x: f"₹{x:,.0f}")
                src_stats['Avg Booking Value'] = src_stats['Avg Booking Value'].apply(lambda x: f"₹{x:,.0f}")
                st.dataframe(src_stats.sort_values('Bookings', ascending=False).reset_index(drop=True), use_container_width=True, hide_index=True)
                abox("💡 <b>Direct bookings</b> are most profitable — no OTA commission (15-25%). Focus on driving direct traffic.", "blue")

    with tab4:
        sh("💰","Profitability Analysis")
        kpi_row([
            (f"₹{total_rev:,.0f}", "Total Revenue", "green", None),
            (f"₹{total_cost:,.0f}", "Total Operating Cost", "red", None),
            (f"₹{net_profit:,.0f}", "Net Profit", "green" if net_profit>=0 else "red", f"{net_profit/total_rev*100:.1f}% margin" if total_rev>0 else None),
        ])
        if staff_col and opcost_col:
            cost_breakdown = {'Staff Cost': nc(df_hot[staff_col]).sum(), 'Operating Cost': nc(df_hot[opcost_col]).sum()}
            fig, ax = plt.subplots(figsize=(6,4))
            clean_chart(fig, ax)
            ax.bar(list(cost_breakdown.keys()), list(cost_breakdown.values()), color=['#6366f1','#ef4444'], width=0.4)
            ax.set_title("Cost Breakdown", fontweight='bold')
            for i,(k,v) in enumerate(cost_breakdown.items()):
                ax.text(i, v, f'₹{v:,.0f}', ha='center', va='bottom', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab5:
        hot_alerts = []
        if avg_occ < 50: hot_alerts.append(('red', f"🔴 Occupancy rate <b>{avg_occ:.1f}%</b> is critically low — run promotions to fill rooms."))
        elif avg_occ < 70: hot_alerts.append(('amber', f"⚠️ Occupancy <b>{avg_occ:.1f}%</b> below 70% target — boost marketing spend."))
        if cancel_rate > 15: hot_alerts.append(('red', f"🔴 Cancellation rate <b>{cancel_rate:.1f}%</b> is very high — review booking policy."))
        if avg_rating < 3.5 and avg_rating > 0: hot_alerts.append(('red', f"🔴 Guest rating <b>{avg_rating:.1f}/5</b> is low — immediate service quality review."))
        if net_profit < 0: hot_alerts.append(('red', f"🔴 Hotel is running at a <b>loss of ₹{abs(net_profit):,.0f}</b> — review cost structure."))
        if not hot_alerts: abox("✅ Hotel operations look healthy — no critical alerts.", "green")
        else:
            for k,m in hot_alerts: abox(m, k)
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# MODULE: AGRICULTURE & FARMING INTELLIGENCE
# ════════════════════════════════════════════════════════════════════════════
if module == "Agriculture":
    df_agr = df.copy()
    show_clean_report(fixes, df_agr, df_raw)

    date_col    = next((c for c in df_agr.columns if 'date' in c.lower()), None)
    crop_col    = next((c for c in df_agr.columns if 'crop' in c.lower()), None)
    season_col  = next((c for c in df_agr.columns if 'season' in c.lower()), None)
    state_col   = next((c for c in df_agr.columns if 'state' in c.lower()), None)
    area_col    = next((c for c in df_agr.columns if 'area' in c.lower()), None)
    exp_yield_col = next((c for c in df_agr.columns if 'expected yield' in c.lower()), None)
    act_yield_col = next((c for c in df_agr.columns if 'actual yield' in c.lower()), None)
    price_col   = next((c for c in df_agr.columns if 'market price' in c.lower()), None)
    msp_col     = next((c for c in df_agr.columns if 'msp' in c.lower()), None)
    rev_col     = next((c for c in df_agr.columns if 'revenue' in c.lower()), None)
    cost_col    = next((c for c in df_agr.columns if 'total cost' in c.lower()), None)
    profit_col  = next((c for c in df_agr.columns if 'net profit' in c.lower() or 'profit' in c.lower()), None)
    seed_col    = next((c for c in df_agr.columns if 'seed' in c.lower()), None)
    fert_col    = next((c for c in df_agr.columns if 'fertilizer' in c.lower()), None)
    labour_col  = next((c for c in df_agr.columns if 'labour' in c.lower()), None)
    irrig_col   = next((c for c in df_agr.columns if 'irrigation' in c.lower()), None)
    market_col  = next((c for c in df_agr.columns if 'market' in c.lower() and 'price' not in c.lower()), None)
    rain_col    = next((c for c in df_agr.columns if 'rainfall' in c.lower()), None)

    if date_col: df_agr[date_col] = pd.to_datetime(df_agr[date_col], dayfirst=True, errors='coerce')
    if exp_yield_col and act_yield_col:
        df_agr['Yield Efficiency %'] = (df_agr[act_yield_col]/df_agr[exp_yield_col]*100).round(1)

    total_rev    = nc(df_agr[rev_col]).sum() if rev_col else 0
    total_cost   = nc(df_agr[cost_col]).sum() if cost_col else 0
    total_profit = nc(df_agr[profit_col]).sum() if profit_col else total_rev - total_cost
    profit_margin= round(total_profit/total_rev*100,1) if total_rev > 0 else 0
    yield_eff    = df_agr['Yield Efficiency %'].mean() if 'Yield Efficiency %' in df_agr.columns else 0
    total_area   = nc(df_agr[area_col]).sum() if area_col else 0
    num_crops    = df_agr[crop_col].nunique() if crop_col else 0

    kpi_row([
        (f"₹{total_rev:,.0f}", "Total Revenue", "indigo", None),
        (f"₹{total_profit:,.0f}", "Net Profit", "green" if total_profit>=0 else "red", f"{profit_margin:.1f}% margin"),
        (f"{yield_eff:.1f}%", "Yield Efficiency", "green" if yield_eff>=90 else "amber" if yield_eff>=75 else "red", "Actual vs Expected"),
        (f"{total_area:.1f}", "Total Acres", "indigo", "Under cultivation"),
        (f"{num_crops}", "Crop Types", "indigo", None),
        (f"₹{total_cost/total_area:.0f}" if total_area > 0 else "N/A", "Cost per Acre", "indigo", "Input efficiency"),
    ])

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌾 Crop Performance", "💰 Profit Analysis", "🌍 State & Market",
        "🧪 Input Cost Analysis", "🚨 Smart Alerts"
    ])

    with tab1:
        sh("🌾","Crop Yield Analysis")
        if crop_col and act_yield_col:
            col_a, col_b = st.columns(2)
            with col_a:
                crop_yield = df_agr.groupby(crop_col)[act_yield_col].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(6,5))
                clean_chart(fig, ax)
                crop_yield.plot(kind='barh', ax=ax, color='#10b981')
                ax.invert_yaxis()
                ax.set_title("Total Yield by Crop (Quintals)", fontweight='bold')
                for i,val in enumerate(crop_yield.values):
                    ax.text(val,i,f'  {val:.0f}Q',va='center',fontsize=9)
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                if 'Yield Efficiency %' in df_agr.columns:
                    crop_eff = df_agr.groupby(crop_col)['Yield Efficiency %'].mean().sort_values()
                    fig, ax = plt.subplots(figsize=(6,5))
                    clean_chart(fig, ax)
                    bar_c = ['#ef4444' if v<75 else '#f59e0b' if v<90 else '#10b981' for v in crop_eff.values]
                    ax.barh(crop_eff.index, crop_eff.values, color=bar_c)
                    ax.axvline(90, color='#6366f1', linestyle='--', linewidth=1.5, label='Target 90%')
                    ax.set_title("Yield Efficiency by Crop", fontweight='bold')
                    ax.legend(fontsize=9)
                    for i,val in enumerate(crop_eff.values):
                        ax.text(val+0.2,i,f'{val:.0f}%',va='center',fontsize=9)
                    plt.tight_layout(); st.pyplot(fig); plt.close()

        if season_col and act_yield_col:
            sh("📅","Yield by Season")
            season_yield = df_agr.groupby(season_col)[act_yield_col].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(6,3))
            clean_chart(fig, ax)
            ax.bar(season_yield.index, season_yield.values, color=['#f59e0b','#6366f1','#10b981'][:len(season_yield)], width=0.4)
            ax.set_title("Total Yield by Season", fontweight='bold')
            for i,(idx,val) in enumerate(season_yield.items()):
                ax.text(i, val, f'{val:.0f}Q', ha='center', va='bottom', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        sh("💰","Profitability by Crop")
        if crop_col and profit_col:
            col_a, col_b = st.columns(2)
            with col_a:
                crop_profit = df_agr.groupby(crop_col)[profit_col].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(6,5))
                clean_chart(fig, ax)
                bar_c = ['#10b981' if v>=0 else '#ef4444' for v in crop_profit.values]
                crop_profit.plot(kind='barh', ax=ax, color=bar_c)
                ax.invert_yaxis()
                ax.axvline(0, color='#6b7280', linestyle='-', linewidth=1)
                ax.set_title("Net Profit by Crop", fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                crop_summary = df_agr.groupby(crop_col).agg({
                    rev_col:'sum', cost_col:'sum', profit_col:'sum'
                }).reset_index() if rev_col and cost_col and profit_col else df_agr.groupby(crop_col)[profit_col].sum().reset_index()
                if rev_col and cost_col and profit_col:
                    crop_summary.columns = ['Crop','Revenue','Total Cost','Net Profit']
                    crop_summary['Margin %'] = (crop_summary['Net Profit']/crop_summary['Revenue']*100).round(1)
                    crop_summary = crop_summary.sort_values('Net Profit', ascending=False)
                    st.dataframe(crop_summary.style.background_gradient(subset=['Margin %'], cmap='RdYlGn'), use_container_width=True, hide_index=True)

        if price_col and msp_col and crop_col:
            sh("📊","Market Price vs MSP")
            price_comp = df_agr.groupby(crop_col).agg({price_col:'mean', msp_col:'mean'}).reset_index()
            price_comp.columns = ['Crop','Market Price','MSP']
            price_comp['Above MSP'] = price_comp['Market Price'] >= price_comp['MSP']
            fig, ax = plt.subplots(figsize=(10,4))
            clean_chart(fig, ax)
            x = range(len(price_comp))
            ax.bar([i-0.2 for i in x], price_comp['MSP'], 0.35, label='MSP', color='#e0e7ff', edgecolor='white')
            ax.bar([i+0.2 for i in x], price_comp['Market Price'], 0.35, label='Market Price',
                   color=['#10b981' if v else '#ef4444' for v in price_comp['Above MSP']], edgecolor='white')
            ax.set_xticks(list(x)); ax.set_xticklabels(price_comp['Crop'], rotation=30, ha='right')
            ax.legend(); ax.set_title("Market Price vs MSP by Crop", fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
            below_msp = price_comp[~price_comp['Above MSP']]
            if len(below_msp) > 0:
                abox(f"⚠️ <b>{len(below_msp)} crops</b> are selling below MSP — consider waiting or exploring better markets.", "amber")

    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            if state_col and rev_col:
                sh("🌍","Revenue by State")
                state_rev = df_agr.groupby(state_col)[rev_col].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                ax.barh(state_rev.index, state_rev.values, color='#6366f1')
                ax.invert_yaxis()
                ax.set_title("Revenue by State", fontweight='bold')
                for i,val in enumerate(state_rev.values):
                    ax.text(val,i,f'  ₹{val:,.0f}',va='center',fontsize=9)
                plt.tight_layout(); st.pyplot(fig); plt.close()
        with col_b:
            if market_col and rev_col:
                sh("🏪","Best Markets")
                mkt_rev = df_agr.groupby(market_col)[rev_col].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                ax.pie(mkt_rev.values, labels=mkt_rev.index, autopct='%1.1f%%',
                       colors=['#6366f1','#818cf8','#a5b4fc','#c7d2fe','#10b981'],
                       wedgeprops=dict(edgecolor='white',linewidth=2))
                ax.set_title("Revenue by Market Type", fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab4:
        sh("🧪","Input Cost Breakdown")
        cost_items = {}
        if seed_col: cost_items['Seeds'] = nc(df_agr[seed_col]).sum()
        if fert_col: cost_items['Fertilizer'] = nc(df_agr[fert_col]).sum()
        if labour_col: cost_items['Labour'] = nc(df_agr[labour_col]).sum()
        if irrig_col: cost_items['Irrigation'] = nc(df_agr[irrig_col]).sum()
        if cost_items:
            col_a, col_b = st.columns(2)
            with col_a:
                fig, ax = plt.subplots(figsize=(5,5))
                clean_chart(fig, ax)
                ax.pie(list(cost_items.values()), labels=list(cost_items.keys()), autopct='%1.1f%%',
                       colors=['#6366f1','#10b981','#f59e0b','#818cf8'],
                       wedgeprops=dict(edgecolor='white',linewidth=2))
                ax.set_title("Input Cost Breakdown", fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                for k,v in sorted(cost_items.items(), key=lambda x:-x[1]):
                    pct = v/sum(cost_items.values())*100
                    st.markdown(f"<div style='display:flex;justify-content:space-between;background:white;border-radius:10px;padding:12px 16px;margin:5px 0;border-left:4px solid #6366f1;box-shadow:0 1px 4px rgba(0,0,0,0.05);'><span style='font-weight:600'>{k}</span><span style='font-weight:800;color:#6366f1'>₹{v:,.0f} <span style='font-size:0.75rem;color:#6b7280'>({pct:.1f}%)</span></span></div>", unsafe_allow_html=True)

    with tab5:
        agr_alerts = []
        if yield_eff < 75: agr_alerts.append(('red', f"🔴 Yield efficiency <b>{yield_eff:.1f}%</b> is critically low — review soil quality, seeds, and irrigation."))
        elif yield_eff < 90: agr_alerts.append(('amber', f"⚠️ Yield efficiency <b>{yield_eff:.1f}%</b> below 90% target."))
        if total_profit < 0: agr_alerts.append(('red', f"🔴 Farming is running at a <b>loss of ₹{abs(total_profit):,.0f}</b> — review input costs and crop selection."))
        if profit_margin < 15: agr_alerts.append(('amber', f"⚠️ Profit margin only <b>{profit_margin:.1f}%</b> — diversify crops or reduce input costs."))
        if price_col and msp_col:
            below = (df_agr[price_col] < df_agr[msp_col]).sum()
            if below > 0: agr_alerts.append(('amber', f"⚠️ <b>{below} records</b> show selling below MSP — explore better market channels."))
        if not agr_alerts: abox("✅ Farm operations look healthy — no critical alerts.", "green")
        else:
            for k,m in agr_alerts: abox(m, k)
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# MODULE: CONSTRUCTION & PROJECTS INTELLIGENCE
# ════════════════════════════════════════════════════════════════════════════
if module == "Construction":
    df_con = df.copy()
    show_clean_report(fixes, df_con, df_raw)

    date_col    = next((c for c in df_con.columns if 'date' in c.lower()), None)
    proj_col    = next((c for c in df_con.columns if 'project' in c.lower()), None)
    phase_col   = next((c for c in df_con.columns if 'phase' in c.lower()), None)
    cont_col    = next((c for c in df_con.columns if 'contractor' in c.lower()), None)
    budget_col  = next((c for c in df_con.columns if 'budget' in c.lower()), None)
    actual_col  = next((c for c in df_con.columns if 'actual cost' in c.lower()), None)
    plan_pct    = next((c for c in df_con.columns if 'planned progress' in c.lower()), None)
    act_pct     = next((c for c in df_con.columns if 'actual progress' in c.lower()), None)
    delay_col   = next((c for c in df_con.columns if 'delay' in c.lower()), None)
    incident_col= next((c for c in df_con.columns if 'incident' in c.lower() or 'safety' in c.lower()), None)
    quality_col = next((c for c in df_con.columns if 'quality' in c.lower()), None)
    workers_col = next((c for c in df_con.columns if 'worker' in c.lower()), None)

    if date_col: df_con[date_col] = pd.to_datetime(df_con[date_col], dayfirst=True, errors='coerce')
    if budget_col and actual_col:
        df_con['Cost Variance'] = df_con[actual_col] - df_con[budget_col]
        df_con['Over Budget'] = df_con['Cost Variance'] > 0
    if plan_pct and act_pct:
        df_con['Progress Gap'] = df_con[plan_pct] - df_con[act_pct]

    total_budget  = nc(df_con[budget_col]).sum() if budget_col else 0
    total_actual  = nc(df_con[actual_col]).sum() if actual_col else 0
    cost_overrun  = total_actual - total_budget
    over_pct      = cost_overrun/total_budget*100 if total_budget > 0 else 0
    avg_progress  = df_con[act_pct].mean() if act_pct else 0
    total_delay   = nc(df_con[delay_col]).sum() if delay_col else 0
    total_incidents = nc(df_con[incident_col]).sum() if incident_col else 0
    avg_quality   = nc(df_con[quality_col]).mean() or 0 if quality_col else 0
    num_projects  = df_con[proj_col].nunique() if proj_col else 0

    kpi_row([
        (f"{num_projects}", "Active Projects", "indigo", None),
        (f"₹{total_budget:,.0f}", "Total Budget", "indigo", None),
        (f"₹{abs(cost_overrun):,.0f}", "Cost Overrun", "red" if cost_overrun>0 else "green", f"{'Over' if cost_overrun>0 else 'Under'} budget {abs(over_pct):.1f}%"),
        (f"{avg_progress:.1f}%", "Avg Progress", "green" if avg_progress>=80 else "amber" if avg_progress>=50 else "red", "Actual completion"),
        (f"{total_delay:.0f} days", "Total Delays", "red" if total_delay>100 else "amber" if total_delay>30 else "green", "Across all projects"),
        (f"{total_incidents:.0f}", "Safety Incidents", "red" if total_incidents>10 else "amber" if total_incidents>5 else "green", "Target: 0"),
    ])

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Project Overview","💰 Budget vs Actual","⏱️ Progress & Delays","🏗️ Contractor Analysis","🚨 Smart Alerts"
    ])

    with tab1:
        if proj_col and act_pct:
            sh("📊","Project Progress Status")
            proj_progress = df_con.groupby(proj_col)[act_pct].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10,5))
            clean_chart(fig, ax)
            bar_c = ['#10b981' if v>=80 else '#f59e0b' if v>=50 else '#ef4444' for v in proj_progress.values]
            ax.barh(proj_progress.index, proj_progress.values, color=bar_c, height=0.5)
            ax.axvline(80, color='#6366f1', linestyle='--', linewidth=1.5, label='Target 80%')
            ax.set_title("Actual Progress % by Project", fontweight='bold')
            ax.set_xlabel("Progress %")
            ax.legend(fontsize=9)
            for i,val in enumerate(proj_progress.values):
                ax.text(val+0.3, i, f'{val:.0f}%', va='center', fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        if phase_col:
            sh("🔄","Work by Phase")
            phase_count = df_con[phase_col].value_counts()
            fig, ax = plt.subplots(figsize=(8,3))
            clean_chart(fig, ax)
            ax.bar(phase_count.index, phase_count.values, color='#818cf8', width=0.5)
            ax.set_title("Activity Count by Phase", fontweight='bold')
            for i,(idx,val) in enumerate(phase_count.items()):
                ax.text(i, val+0.2, str(val), ha='center', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        sh("💰","Budget vs Actual Cost")
        if proj_col and budget_col and actual_col:
            proj_cost = df_con.groupby(proj_col).agg({budget_col:'sum', actual_col:'sum'}).reset_index()
            proj_cost.columns = ['Project','Budget','Actual']
            proj_cost['Variance'] = proj_cost['Actual'] - proj_cost['Budget']
            proj_cost['Over Budget'] = proj_cost['Variance'] > 0
            fig, ax = plt.subplots(figsize=(10,5))
            clean_chart(fig, ax)
            x = range(len(proj_cost))
            ax.bar([i-0.2 for i in x], proj_cost['Budget']/1000, 0.35, label='Budget (₹K)', color='#e0e7ff', edgecolor='white')
            ax.bar([i+0.2 for i in x], proj_cost['Actual']/1000, 0.35, label='Actual (₹K)',
                   color=['#ef4444' if v else '#10b981' for v in proj_cost['Over Budget']], edgecolor='white')
            ax.set_xticks(list(x)); ax.set_xticklabels(proj_cost['Project'], rotation=30, ha='right', fontsize=8)
            ax.legend(); ax.set_title("Budget vs Actual by Project", fontweight='bold')
            ax.set_ylabel("Amount (₹ Thousands)")
            plt.tight_layout(); st.pyplot(fig); plt.close()

            over_projects = proj_cost[proj_cost['Over Budget']]
            if len(over_projects) > 0:
                abox(f"⚠️ <b>{len(over_projects)} projects</b> are over budget — total overrun ₹{over_projects['Variance'].sum():,.0f}", "amber")
                st.dataframe(over_projects[['Project','Budget','Actual','Variance']].reset_index(drop=True), use_container_width=True, hide_index=True)

    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            if proj_col and delay_col:
                sh("⏱️","Delays by Project")
                proj_delay = df_con.groupby(proj_col)[delay_col].sum().sort_values(ascending=False).head(8)
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                ax.barh(proj_delay.index, proj_delay.values, color='#ef4444', alpha=0.85)
                ax.invert_yaxis()
                ax.set_title("Total Delay Days by Project", fontweight='bold')
                for i,val in enumerate(proj_delay.values):
                    ax.text(val+0.2, i, f'{val:.0f}d', va='center', fontsize=9)
                plt.tight_layout(); st.pyplot(fig); plt.close()
        with col_b:
            if plan_pct and act_pct and proj_col:
                sh("📉","Progress Gap Analysis")
                gap_data = df_con.groupby(proj_col).agg({plan_pct:'mean', act_pct:'mean'}).reset_index()
                gap_data.columns = ['Project','Planned %','Actual %']
                gap_data['Gap'] = gap_data['Planned %'] - gap_data['Actual %']
                behind = gap_data[gap_data['Gap'] > 10]
                if len(behind) > 0:
                    abox(f"⚠️ <b>{len(behind)} projects</b> are significantly behind schedule.", "amber")
                    st.dataframe(behind.sort_values('Gap', ascending=False).reset_index(drop=True), use_container_width=True, hide_index=True)
                else:
                    abox("✅ All projects are on or near schedule.", "green")

        if incident_col and proj_col:
            sh("⚠️","Safety Incidents")
            inc_by_proj = df_con.groupby(proj_col)[incident_col].sum().sort_values(ascending=False)
            if inc_by_proj.sum() > 0:
                fig, ax = plt.subplots(figsize=(10,3))
                clean_chart(fig, ax)
                bar_c = ['#ef4444' if v>3 else '#f59e0b' if v>0 else '#10b981' for v in inc_by_proj.values]
                ax.bar(inc_by_proj.index, inc_by_proj.values, color=bar_c, width=0.5)
                ax.set_title("Safety Incidents by Project", fontweight='bold')
                plt.xticks(rotation=30, ha='right', fontsize=9)
                plt.tight_layout(); st.pyplot(fig); plt.close()
            else:
                abox("✅ Zero safety incidents recorded — excellent safety record.", "green")

    with tab4:
        if cont_col and actual_col:
            sh("🏗️","Contractor Performance")
            cont_stats = df_con.groupby(cont_col).agg({
                actual_col:'sum', budget_col:'sum'
            }).reset_index() if budget_col else df_con.groupby(cont_col)[actual_col].sum().reset_index()
            if budget_col:
                cont_stats.columns = ['Contractor','Actual Cost','Budget']
                cont_stats['Cost Efficiency %'] = (cont_stats['Budget']/cont_stats['Actual Cost']*100).round(1)
            if quality_col:
                cont_quality = df_con.groupby(cont_col)[quality_col].mean().reset_index()
                cont_quality.columns = ['Contractor','Avg Quality Score']
                cont_stats = cont_stats.merge(cont_quality, on='Contractor')
            if delay_col:
                cont_delay = df_con.groupby(cont_col)[delay_col].sum().reset_index()
                cont_delay.columns = ['Contractor','Total Delays']
                cont_stats = cont_stats.merge(cont_delay, on='Contractor')
            cont_stats = cont_stats.sort_values('Actual Cost', ascending=False).reset_index(drop=True)
            st.dataframe(cont_stats, use_container_width=True, hide_index=True)

    with tab5:
        con_alerts = []
        if cost_overrun > total_budget*0.1: con_alerts.append(('red', f"🔴 Total cost overrun is <b>₹{cost_overrun:,.0f} ({over_pct:.1f}%)</b> — review project spending urgently."))
        elif cost_overrun > 0: con_alerts.append(('amber', f"⚠️ Projects are <b>₹{cost_overrun:,.0f}</b> over budget — monitor closely."))
        if total_delay > 100: con_alerts.append(('red', f"🔴 <b>{total_delay:.0f} days</b> of delays across projects — review timeline management."))
        if total_incidents > 5: con_alerts.append(('red', f"🔴 <b>{total_incidents:.0f} safety incidents</b> recorded — immediate safety review required."))
        if avg_quality < 75 and avg_quality > 0: con_alerts.append(('amber', f"⚠️ Average quality score <b>{avg_quality:.1f}</b> is below 75 — review contractor standards."))
        if not con_alerts: abox("✅ Construction projects look on track — no critical alerts.", "green")
        else:
            for k,m in con_alerts: abox(m, k)
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# MODULE: BANKING & MICROFINANCE INTELLIGENCE
# ════════════════════════════════════════════════════════════════════════════
if module == "Banking":
    df_bnk = df.copy()
    show_clean_report(fixes, df_bnk, df_raw)

    date_col   = next((c for c in df_bnk.columns if 'date' in c.lower()), None)
    type_col   = next((c for c in df_bnk.columns if 'loan type' in c.lower()), None)
    branch_col = next((c for c in df_bnk.columns if 'branch' in c.lower()), None)
    agent_col  = next((c for c in df_bnk.columns if 'agent' in c.lower()), None)
    amt_col    = next((c for c in df_bnk.columns if 'loan amount' in c.lower()), None)
    rate_col   = next((c for c in df_bnk.columns if 'interest rate' in c.lower()), None)
    emi_col    = next((c for c in df_bnk.columns if 'emi' in c.lower()), None)
    paid_col   = next((c for c in df_bnk.columns if 'emis paid' in c.lower()), None)
    out_col    = next((c for c in df_bnk.columns if 'outstanding' in c.lower()), None)
    status_col = next((c for c in df_bnk.columns if 'status' in c.lower()), None)
    overdue_col= next((c for c in df_bnk.columns if 'overdue' in c.lower()), None)
    collect_col= next((c for c in df_bnk.columns if 'collection %' in c.lower()), None)
    state_col  = next((c for c in df_bnk.columns if 'state' in c.lower()), None)

    if date_col: df_bnk[date_col] = pd.to_datetime(df_bnk[date_col], dayfirst=True, errors='coerce')

    total_disbursed = nc(df_bnk[amt_col]).sum() if amt_col else 0
    total_outstanding= nc(df_bnk[out_col]).sum() if out_col else 0
    npa_count = (df_bnk[status_col].astype(str).str.upper()=='NPA').sum() if status_col else 0
    npa_rate  = npa_count/len(df_bnk)*100
    npa_value = df_bnk[df_bnk[status_col].astype(str).str.upper()=='NPA'][out_col].sum() if status_col and out_col else 0
    avg_collect = nc(df_bnk[collect_col]).mean() or 0 if collect_col else 0
    avg_rate  = nc(df_bnk[rate_col]).mean() or 0 if rate_col else 0
    active_loans = (df_bnk[status_col].astype(str).str.title()=='Active').sum() if status_col else len(df_bnk)

    kpi_row([
        (f"₹{total_disbursed/1e6:.1f}M", "Total Disbursed", "indigo", f"{len(df_bnk):,} loans"),
        (f"₹{total_outstanding/1e6:.1f}M", "Total Outstanding", "indigo", "Portfolio size"),
        (f"{npa_rate:.1f}%", "NPA Rate", "red" if npa_rate>10 else "amber" if npa_rate>5 else "green", "Target: <5%"),
        (f"₹{npa_value/1e6:.1f}M", "NPA Value", "red" if npa_value>0 else "green", "At risk"),
        (f"{avg_collect:.1f}%", "Avg Collection %", "green" if avg_collect>=85 else "amber" if avg_collect>=70 else "red", "Recovery rate"),
        (f"{active_loans:,}", "Active Loans", "indigo", None),
    ])

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Portfolio Overview","⚠️ NPA Analysis","🏦 Branch Performance","👤 Agent Performance","🚨 Smart Alerts"
    ])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            if type_col and amt_col:
                sh("📊","Loan Portfolio by Type")
                type_amt = df_bnk.groupby(type_col)[amt_col].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(6,5))
                clean_chart(fig, ax)
                ax.pie(type_amt.values, labels=type_amt.index, autopct='%1.1f%%',
                       colors=['#6366f1','#818cf8','#a5b4fc','#c7d2fe','#e0e7ff','#4f46e5','#10b981','#34d399'],
                       wedgeprops=dict(edgecolor='white',linewidth=2))
                ax.set_title("Portfolio by Loan Type", fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
        with col_b:
            if status_col:
                sh("📋","Loan Status Breakdown")
                status_count = df_bnk[status_col].value_counts()
                fig, ax = plt.subplots(figsize=(6,5))
                clean_chart(fig, ax)
                status_colors = {'Active':'#10b981','Closed':'#6366f1','NPA':'#ef4444','Written Off':'#9ca3af','Restructured':'#f59e0b'}
                bar_c = [status_colors.get(s,'#6366f1') for s in status_count.index]
                ax.pie(status_count.values, labels=status_count.index, autopct='%1.1f%%',
                       colors=bar_c, wedgeprops=dict(edgecolor='white',linewidth=2))
                ax.set_title("Portfolio Status Split", fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()

        if type_col and amt_col and out_col:
            sh("💰","Outstanding by Loan Type")
            type_out = df_bnk.groupby(type_col).agg({amt_col:'sum', out_col:'sum'}).reset_index()
            type_out.columns = ['Loan Type','Disbursed','Outstanding']
            type_out['Collection %'] = ((type_out['Disbursed']-type_out['Outstanding'])/type_out['Disbursed']*100).round(1)
            type_out = type_out.sort_values('Outstanding', ascending=False)
            st.dataframe(type_out.style.background_gradient(subset=['Collection %'], cmap='RdYlGn'), use_container_width=True, hide_index=True)

    with tab2:
        sh("⚠️","NPA Analysis")
        if status_col and out_col:
            npa_df = df_bnk[df_bnk[status_col].astype(str).str.upper()=='NPA'].copy()
            kpi_row([
                (f"{len(npa_df)}", "NPA Accounts", "red", None),
                (f"₹{npa_value:,.0f}", "NPA Outstanding", "red", "At risk"),
                (f"{npa_rate:.1f}%", "NPA Rate", "red" if npa_rate>10 else "amber", "of portfolio"),
            ])
            if len(npa_df) > 0:
                abox(f"🔴 <b>{len(npa_df)} accounts</b> are NPA with ₹{npa_value:,.0f} outstanding — initiate recovery process immediately.", "red")
                if type_col:
                    npa_by_type = npa_df.groupby(type_col)[out_col].sum().sort_values(ascending=False)
                    col_a, col_b = st.columns(2)
                    with col_a:
                        fig, ax = plt.subplots(figsize=(6,4))
                        clean_chart(fig, ax)
                        ax.barh(npa_by_type.index, npa_by_type.values, color='#ef4444', alpha=0.85)
                        ax.invert_yaxis()
                        ax.set_title("NPA Outstanding by Loan Type", fontweight='bold')
                        for i,val in enumerate(npa_by_type.values):
                            ax.text(val,i,f'  ₹{val:,.0f}',va='center',fontsize=9)
                        plt.tight_layout(); st.pyplot(fig); plt.close()
                    with col_b:
                        if overdue_col:
                            show_cols = [c for c in [type_col, branch_col, amt_col, out_col, overdue_col] if c and c in npa_df.columns]
                            st.dataframe(npa_df[show_cols].sort_values(out_col, ascending=False).reset_index(drop=True), use_container_width=True, hide_index=True)

    with tab3:
        if branch_col and amt_col:
            sh("🏦","Branch Performance")
            branch_stats = df_bnk.groupby(branch_col).agg({amt_col:'sum', out_col:'sum'}).reset_index() if out_col else df_bnk.groupby(branch_col)[amt_col].sum().reset_index()
            if out_col:
                branch_stats.columns = ['Branch','Disbursed','Outstanding']
                if status_col:
                    branch_npa = df_bnk[df_bnk[status_col].astype(str).str.upper()=='NPA'].groupby(branch_col).size().reset_index()
                    branch_npa.columns = ['Branch','NPA Count']
                    branch_stats = branch_stats.merge(branch_npa, on='Branch', how='left').fillna(0)
                    branch_stats['NPA Count'] = branch_stats['NPA Count'].astype(int)
                branch_stats['Collection %'] = ((branch_stats['Disbursed']-branch_stats['Outstanding'])/branch_stats['Disbursed']*100).round(1)
            col_a, col_b = st.columns(2)
            with col_a:
                fig, ax = plt.subplots(figsize=(6,5))
                clean_chart(fig, ax)
                branch_disburse = branch_stats.sort_values('Disbursed', ascending=False)
                ax.barh(branch_disburse['Branch'], branch_disburse['Disbursed'], color='#6366f1')
                ax.invert_yaxis()
                ax.set_title("Loans Disbursed by Branch", fontweight='bold')
                for i,val in enumerate(branch_disburse['Disbursed']):
                    ax.text(val,i,f'  ₹{val:,.0f}',va='center',fontsize=8)
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                st.dataframe(branch_stats.sort_values('Disbursed', ascending=False).reset_index(drop=True), use_container_width=True, hide_index=True)

    with tab4:
        if agent_col and amt_col:
            sh("👤","Agent Performance")
            agent_stats = df_bnk.groupby(agent_col).agg(Loans=(amt_col,'count'), Disbursed=(amt_col,'sum')).reset_index()
            if status_col:
                agent_npa = df_bnk[df_bnk[status_col].astype(str).str.upper()=='NPA'].groupby(agent_col).size().reset_index()
                agent_npa.columns = [agent_col,'NPA Count']
                agent_stats = agent_stats.merge(agent_npa, on=agent_col, how='left').fillna(0)
                agent_stats['NPA Count'] = agent_stats['NPA Count'].astype(int)
                agent_stats['NPA Rate %'] = (agent_stats['NPA Count']/agent_stats['Loans']*100).round(1)
            agent_stats = agent_stats.sort_values('Disbursed', ascending=False).reset_index(drop=True)
            col_a, col_b = st.columns(2)
            with col_a:
                fig, ax = plt.subplots(figsize=(6,5))
                clean_chart(fig, ax)
                ax.barh(agent_stats[agent_col].head(10), agent_stats['Disbursed'].head(10), color='#818cf8')
                ax.invert_yaxis()
                ax.set_title("Top 10 Agents by Disbursement", fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_b:
                st.dataframe(agent_stats.style.background_gradient(subset=['NPA Rate %'] if 'NPA Rate %' in agent_stats.columns else ['Loans'], cmap='RdYlGn_r'), use_container_width=True, hide_index=True)

    with tab5:
        bnk_alerts = []
        if npa_rate > 10: bnk_alerts.append(('red', f"🔴 NPA rate <b>{npa_rate:.1f}%</b> is critically high — immediate recovery action required."))
        elif npa_rate > 5: bnk_alerts.append(('amber', f"⚠️ NPA rate <b>{npa_rate:.1f}%</b> exceeds 5% threshold — strengthen credit assessment."))
        if avg_collect < 70: bnk_alerts.append(('red', f"🔴 Collection efficiency only <b>{avg_collect:.1f}%</b> — recovery processes need urgent attention."))
        elif avg_collect < 85: bnk_alerts.append(('amber', f"⚠️ Collection rate <b>{avg_collect:.1f}%</b> below 85% target."))
        if npa_value > total_disbursed*0.1: bnk_alerts.append(('red', f"🔴 NPA value <b>₹{npa_value:,.0f}</b> exceeds 10% of portfolio — systemic risk."))
        if not bnk_alerts: abox("✅ Loan portfolio looks healthy — no critical alerts.", "green")
        else:
            for k,m in bnk_alerts: abox(m, k)
    st.stop()

# ── Sales module uses the already smart-cleaned df ───────────────────────────
df_clean = df.copy()

date_col    = next((c for c in df_clean.columns if 'date' in c.lower()), None)
rev_col     = next((c for c in df_clean.columns if 'total revenue' in c.lower() or ('revenue' in c.lower() and 'unit' not in c.lower())), None)
region_col  = next((c for c in df_clean.columns if 'region' in c.lower()), None)
product_col = next((c for c in df_clean.columns if 'product name' in c.lower()), None)
person_col  = next((c for c in df_clean.columns if 'salesperson' in c.lower()), None)
qty_col     = next((c for c in df_clean.columns if 'quantity' in c.lower() or 'qty' in c.lower()), None)
price_col   = next((c for c in df_clean.columns if 'unit price' in c.lower()), None)
target_col  = next((c for c in df_clean.columns if 'target' in c.lower()), None)
cat_col     = next((c for c in df_clean.columns if 'category' in c.lower()), None)
ctype_col   = next((c for c in df_clean.columns if 'customer type' in c.lower()), None)
payment_col = next((c for c in df_clean.columns if 'payment' in c.lower()), None)
state_col   = next((c for c in df_clean.columns if 'state' in c.lower()), None)

if date_col:
    df_clean[date_col] = pd.to_datetime(df_clean[date_col], dayfirst=True, errors='coerce')

show_clean_report(fixes, df_clean, df_raw)

# Sidebar filters for sales
with st.sidebar:
    st.markdown("### 🔍 Filter Sales Data")
    df_s = df_clean.copy()
    if region_col:
        regions = ['All'] + sorted(df_clean[region_col].dropna().unique().tolist())
        sr = st.selectbox("Region", regions, key="sales_region")
        if sr != 'All': df_s = df_s[df_s[region_col] == sr]
    if cat_col:
        cats = ['All'] + sorted(df_clean[cat_col].dropna().unique().tolist())
        sc = st.selectbox("Category", cats, key="sales_cat")
        if sc != 'All': df_s = df_s[df_s[cat_col] == sc]
    if person_col:
        persons = ['All'] + sorted(df_clean[person_col].dropna().unique().tolist())
        sp = st.selectbox("Salesperson", persons, key="sales_person")
        if sp != 'All': df_s = df_s[df_s[person_col] == sp]
    st.caption(f"**{len(df_s):,}** of **{len(df_clean):,}** rows")

# KPIs
total_rev    = nc(df_s[rev_col]).sum() if rev_col else 0
total_qty    = nc(df_s[qty_col]).sum() if qty_col else 0
total_target = nc(df_s[target_col]).sum() if target_col else 0
achievement  = (total_rev / total_target * 100) if target_col and total_target > 0 else 0

kpi_row([
    (f"₹{total_rev:,.0f}", "Total Revenue", "indigo", None),
    (f"{achievement:.1f}%" if target_col else "N/A", "Target Achievement", "green" if achievement >= 100 else "amber", f"Target: ₹{total_target:,.0f}" if target_col else None),
    (f"{total_qty:,.0f}", "Units Sold", "indigo", None),
    (f"{df_s[product_col].nunique()}" if product_col else "N/A", "Products", "indigo", None),
    (f"{df_s[person_col].nunique()}" if person_col else "N/A", "Salespersons", "indigo", None),
    (f"{df_s[region_col].nunique()}" if region_col else "N/A", "Regions", "indigo", None),
])

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🌍 Revenue","📦 Products","👤 Salesperson","👥 Customers","💳 Payments","🔮 Forecast","🚨 Alerts"
])

with tab1:
    if rev_col:
        col_a, col_b = st.columns(2)
        with col_a:
            if region_col:
                sh("🌍","Revenue by Region")
                rrev = df_s.groupby(region_col)[rev_col].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                bars = ax.bar(rrev.index, rrev.values, color=['#6366f1','#818cf8','#a5b4fc','#c7d2fe'], width=0.5)
                ax.set_ylabel("Revenue (₹)")
                for bar, val in zip(bars, rrev.values):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'₹{val/100000:.1f}L', ha='center', va='bottom', fontsize=9, fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
        with col_b:
            if cat_col:
                sh("📊","Revenue by Category")
                crev = df_s.groupby(cat_col)[rev_col].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                ax.pie(crev.values, labels=crev.index, autopct='%1.1f%%',
                       colors=['#6366f1','#818cf8','#a5b4fc','#c7d2fe','#e0e7ff'],
                       wedgeprops=dict(edgecolor='white', linewidth=2))
                plt.tight_layout(); st.pyplot(fig); plt.close()

        if date_col:
            sh("📈","Revenue Trend")
            monthly = df_s.groupby(df_s[date_col].dt.to_period('M'))[rev_col].sum().reset_index()
            monthly.columns = ['Month','Revenue']
            monthly['Month'] = monthly['Month'].astype(str)
            fig, ax = plt.subplots(figsize=(12,4))
            clean_chart(fig, ax)
            ax.fill_between(range(len(monthly)), monthly['Revenue'], alpha=0.12, color='#6366f1')
            ax.plot(range(len(monthly)), monthly['Revenue'], color='#6366f1', linewidth=2.5, marker='o', markersize=5)
            ax.set_xticks(range(len(monthly)))
            ax.set_xticklabels(monthly['Month'], rotation=45, ha='right', fontsize=8)
            ax.set_title("Monthly Revenue Trend", fontweight='bold')
            ax.set_ylabel("Revenue (₹)")
            plt.tight_layout(); st.pyplot(fig); plt.close()

with tab2:
    if product_col and rev_col:
        col_a, col_b = st.columns(2)
        with col_a:
            sh("🏆","Top 10 Products by Revenue")
            top_p = df_s.groupby(product_col)[rev_col].sum().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(6,5))
            clean_chart(fig, ax)
            top_p.plot(kind='barh', ax=ax, color='#6366f1')
            ax.invert_yaxis()
            for i, val in enumerate(top_p.values):
                ax.text(val, i, f' ₹{val/100000:.1f}L', va='center', fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with col_b:
            if qty_col:
                sh("📦","Top 10 by Units Sold")
                top_q = df_s.groupby(product_col)[qty_col].sum().sort_values(ascending=False).head(10)
                fig, ax = plt.subplots(figsize=(6,5))
                clean_chart(fig, ax)
                top_q.plot(kind='barh', ax=ax, color='#818cf8')
                ax.invert_yaxis()
                plt.tight_layout(); st.pyplot(fig); plt.close()

        if price_col:
            sh("💰","Price Inconsistency Detection")
            price_a = df_s.groupby(product_col)[price_col].agg(['min','max']).reset_index()
            price_a.columns = ['Product','Min Price','Max Price']
            price_a['Variation'] = price_a['Max Price'] - price_a['Min Price']
            incon = price_a[price_a['Variation'] > 0].sort_values('Variation', ascending=False)
            if len(incon) > 0:
                abox(f"⚠️ <b>{len(incon)} products</b> sold at inconsistent prices — same product billed differently!", "amber")
                st.dataframe(incon.reset_index(drop=True), use_container_width=True)
            else:
                abox("✅ Pricing is consistent across all products.", "green")

with tab3:
    if person_col and rev_col:
        sh("👤","Salesperson Performance")
        pdata = df_s.groupby(person_col).agg(Revenue=(rev_col,'sum'), Orders=(rev_col,'count')).reset_index()
        if target_col:
            ptarget = df_s.groupby(person_col)[target_col].sum().reset_index()
            ptarget.columns = [person_col,'Target']
            pdata = pdata.merge(ptarget, on=person_col)
            pdata['Achievement %'] = (pdata['Revenue']/pdata['Target']*100).round(1)
        pdata = pdata.sort_values('Revenue', ascending=False).reset_index(drop=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots(figsize=(7,5))
            clean_chart(fig, ax)
            bar_c = ['#10b981' if (target_col and i < len(pdata) and pdata.loc[i,'Achievement %'] >= 100 if 'Achievement %' in pdata.columns else False) else '#6366f1' for i in range(len(pdata))]
            bars = ax.bar(pdata[person_col], pdata['Revenue'], color='#6366f1', width=0.6)
            ax.set_ylabel("Revenue (₹)")
            plt.xticks(rotation=30, ha='right', fontsize=9)
            for bar, val in zip(bars, pdata['Revenue']):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'₹{val/100000:.1f}L', ha='center', va='bottom', fontsize=8, fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with col_b:
            disp = pdata.copy()
            disp['Revenue'] = disp['Revenue'].apply(lambda x: f"₹{x:,.0f}")
            if 'Target' in disp.columns:
                disp['Target'] = disp['Target'].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(disp, use_container_width=True)

        top_p = pdata.iloc[0][person_col]
        bot_p = pdata.iloc[-1][person_col]
        abox(f"🏆 <b>Top Performer:</b> {top_p}  &nbsp;|&nbsp;  ⚠️ <b>Needs Support:</b> {bot_p}", "blue")

with tab4:
    if ctype_col and rev_col:
        sh("👥","Customer Intelligence")
        col_a, col_b = st.columns(2)
        with col_a:
            crev = df_s.groupby(ctype_col)[rev_col].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(6,4))
            clean_chart(fig, ax)
            ax.pie(crev.values, labels=crev.index, autopct='%1.1f%%',
                   colors=['#6366f1','#818cf8','#a5b4fc','#c7d2fe'],
                   wedgeprops=dict(edgecolor='white', linewidth=2))
            ax.set_title("Revenue by Customer Type", fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with col_b:
            csum = df_s.groupby(ctype_col).agg(Revenue=(rev_col,'sum'), Orders=(rev_col,'count')).reset_index()
            csum['Avg Order'] = (csum['Revenue']/csum['Orders']).round(0)
            csum['Revenue'] = csum['Revenue'].apply(lambda x: f"₹{x:,.0f}")
            csum['Avg Order'] = csum['Avg Order'].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(csum, use_container_width=True)

with tab5:
    if payment_col and rev_col:
        sh("💳","Payment Mode Analysis")
        col_a, col_b = st.columns(2)
        with col_a:
            pay_rev = df_s.groupby(payment_col)[rev_col].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(6,4))
            clean_chart(fig, ax)
            ax.bar(pay_rev.index, pay_rev.values, color=['#6366f1','#818cf8','#a5b4fc','#c7d2fe','#e0e7ff'], width=0.5)
            ax.set_ylabel("Revenue (₹)")
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with col_b:
            psum = df_s.groupby(payment_col).agg(Revenue=(rev_col,'sum'), Transactions=(rev_col,'count')).reset_index()
            psum['Revenue %'] = (psum['Revenue']/psum['Revenue'].sum()*100).round(1)
            psum['Revenue'] = psum['Revenue'].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(psum, use_container_width=True)

with tab6:
    forecast_values = None
    future_months = None
    growth_pct = None
    forecast_buf = None

    if date_col and rev_col:
        sh("🔮","6-Month Sales Forecast")
        monthly = df_s.groupby(df_s[date_col].dt.to_period('M'))[rev_col].sum().reset_index()
        monthly.columns = ['Month','Revenue']
        monthly['MonthNum'] = range(len(monthly))
        X = monthly['MonthNum'].values.reshape(-1,1)
        y = monthly['Revenue'].values
        model = LinearRegression()
        model.fit(X, y)
        future_months = pd.period_range(start=monthly['Month'].iloc[-1]+1, periods=6, freq='M')
        future_nums = np.array(range(len(monthly), len(monthly)+6)).reshape(-1,1)
        forecast_values = np.maximum(model.predict(future_nums), 0)

        fig, ax = plt.subplots(figsize=(12,5))
        clean_chart(fig, ax)
        ax.fill_between(monthly['MonthNum'], monthly['Revenue'], alpha=0.1, color='#6366f1')
        ax.plot(monthly['MonthNum'], monthly['Revenue'], color='#6366f1', linewidth=2.5, marker='o', markersize=5, label='Historical')
        ax.plot(future_nums, forecast_values, color='#f59e0b', linewidth=2.5, linestyle='--', marker='o', markersize=5, label='Forecast')
        ax.fill_between(future_nums.flatten(), forecast_values*0.88, forecast_values*1.12, alpha=0.12, color='#f59e0b', label='Confidence range')
        ax.axvline(x=len(monthly)-0.5, color='#9ca3af', linestyle=':', alpha=0.7)
        ax.set_title("Sales Forecast — Next 6 Months", fontweight='bold', fontsize=13)
        ax.set_ylabel("Revenue (₹)")
        ax.legend(fontsize=10)
        plt.tight_layout(); st.pyplot(fig)

        forecast_buf = io.BytesIO()
        fig.savefig(forecast_buf, format='png', dpi=150, bbox_inches='tight')
        forecast_buf.seek(0)
        plt.close()

        fdf = pd.DataFrame({'Month':[str(m) for m in future_months], 'Forecasted Revenue':[f"₹{v:,.0f}" for v in forecast_values]})
        st.table(fdf)

        last6 = monthly['Revenue'].tail(6).mean()
        next6 = forecast_values.mean()
        growth_pct = ((next6-last6)/last6*100)
        if growth_pct >= 0:
            abox(f"📈 Revenue trending <b>UPWARD</b> — Expected growth: <b>+{growth_pct:.1f}%</b>", "green")
        else:
            abox(f"📉 Revenue trending <b>DOWNWARD</b> — Expected change: <b>{growth_pct:.1f}%</b>. Review strategy now.", "amber")

with tab7:
    sh("🚨","Smart Business Alerts")
    sales_alerts = []
    if growth_pct is not None and growth_pct < -10:
        sales_alerts.append(('red', f"🔴 Revenue declining at <b>{growth_pct:.1f}%</b> — immediate strategy review needed"))
    if product_col and qty_col:
        slow = df_s.groupby(product_col)[qty_col].sum().sort_values().head(3)
        for p, q in slow.items():
            sales_alerts.append(('amber', f"⚠️ Slow moving: <b>{p}</b> — only {q:.0f} units sold"))
    if person_col and rev_col:
        prev = df_s.groupby(person_col)[rev_col].sum()
        avg_r = prev.mean()
        for p in prev[prev < avg_r * 0.7].index:
            sales_alerts.append(('amber', f"⚠️ <b>{p}</b> is 30%+ below team average revenue"))
    if not sales_alerts:
        abox("✅ No critical alerts — business looks healthy!", "green")
    else:
        for kind, msg in sales_alerts:
            abox(msg, kind)

    # PDF Report
    sh("📄","Download Full Report")
    if st.button("🗒️ Generate PDF Report"):
        pdf_buf = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buf, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
        styles = getSampleStyleSheet()
        story = []
        title_style = ParagraphStyle('T', fontSize=22, textColor=colors.HexColor('#6366f1'), spaceAfter=6, alignment=1, fontName='Helvetica-Bold')
        sub_style   = ParagraphStyle('S', fontSize=12, textColor=colors.grey, spaceAfter=4, alignment=1)
        sec_style   = ParagraphStyle('Sec', fontSize=14, textColor=colors.HexColor('#0a0a0a'), spaceBefore=16, spaceAfter=8, fontName='Helvetica-Bold')
        story.append(Paragraph("Velytics", title_style))
        story.append(Paragraph("Sales Intelligence Report", sub_style))
        story.append(Paragraph(f"Generated: {pd.Timestamp.now().strftime('%d %B %Y, %I:%M %p')}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("Key Metrics", sec_style))
        metrics = [["Metric","Value"]]
        if rev_col: metrics.append(["Total Revenue", f"₹{nc(df_s[rev_col]).sum():,.0f}"])
        if target_col: metrics.append(["Target Achievement", f"{achievement:.1f}%"])
        if qty_col: metrics.append(["Total Units Sold", f"{nc(df_s[qty_col]).sum():,.0f}"])
        if region_col: metrics.append(["Top Region", df_s.groupby(region_col)[rev_col].sum().idxmax()])
        if product_col: metrics.append(["Top Product", df_s.groupby(product_col)[rev_col].sum().idxmax()])
        metrics.append(["Records Analyzed", str(len(df_s))])
        t = Table(metrics, colWidths=[2.5*inch, 3.5*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#6366f1')),
            ('TEXTCOLOR',(0,0),(-1,0),colors.white),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.whitesmoke, colors.white]),
            ('GRID',(0,0),(-1,-1),0.5,colors.grey),
            ('PADDING',(0,0),(-1,-1),8),
        ]))
        story.append(t)
        if forecast_buf:
            story.append(Paragraph("Sales Forecast", sec_style))
            forecast_buf.seek(0)
            story.append(RLImage(forecast_buf, width=6*inch, height=3*inch))
        if sales_alerts:
            story.append(Paragraph("Business Alerts", sec_style))
            for _, alert in sales_alerts:
                clean_a = alert.replace("**","").replace("<b>","").replace("</b>","").replace("🔴","WARNING:").replace("⚠️","ALERT:")
                story.append(Paragraph(f"• {clean_a}", styles['Normal']))
                story.append(Spacer(1, 0.05*inch))
        doc.build(story)
        pdf_buf.seek(0)
        st.download_button("📥 Download Velytics Report", data=pdf_buf, file_name="velytics_sales_report.pdf", mime="application/pdf")
        abox("✅ PDF report ready — click above to download!", "green")
