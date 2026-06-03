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

    if best_row > 0:
        fixes.append(f"📋 Detected real headers on row {best_row + 1} — skipped {best_row} title row(s)")
        df.columns = df.iloc[best_row].astype(str).str.strip()
        df = df.iloc[best_row + 1:].reset_index(drop=True)

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
    df.drop_duplicates(inplace=True)
    dups = before - len(df)
    if dups > 0:
        fixes.append(f"🔁 Removed {dups} duplicate row(s)")
    df.reset_index(drop=True, inplace=True)

    # ── 8. CLEAN TEXT COLUMNS ────────────────────────────────────────────────
    text_fixes = 0
    for col in df.select_dtypes(include='object').columns:
        original = df[col].copy()
        # Strip whitespace
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
        # Fix 'nan' strings
        df[col] = df[col].replace({'nan': np.nan, 'none': np.nan, 'None': np.nan,
                                    'NULL': np.nan, 'null': np.nan, 'N/A': np.nan,
                                    'n/a': np.nan, 'NA': np.nan, '': np.nan})
        # Title case for category-like columns (low unique count)
        if df[col].nunique() < 30 and df[col].nunique() > 0:
            df[col] = df[col].str.title()
        changed = (df[col].fillna('') != original.fillna('')).sum()
        text_fixes += changed
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
            numeric_count = pd.to_numeric(cleaned, errors='coerce').notna().sum()
            if numeric_count >= len(sample) * 0.7 and len(sample) > 0:
                df[col] = df[col].astype(str)\
                    .str.replace(r'[₹$€£,\s%]', '', regex=True)\
                    .str.replace(r'^\((.+)\)$', r'-\1', regex=True)
                converted = pd.to_numeric(df[col], errors='coerce')
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
            df[col].fillna(df[col].median(), inplace=True)
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


def load_file(uploaded_file):
    """Load any Excel/CSV — including multi-sheet Excel files."""
    name = uploaded_file.name.lower()
    if name.endswith('.csv'):
        try:
            return pd.read_csv(uploaded_file)
        except:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding='latin-1')
    else:
        xl = pd.ExcelFile(uploaded_file)
        sheets = xl.sheet_names
        if len(sheets) == 1:
            return pd.read_excel(xl, sheet_name=sheets[0], header=None)
        else:
            # Multiple sheets — let user pick or auto-merge
            if 'selected_sheet' not in st.session_state:
                st.session_state.selected_sheet = sheets[0]
            selected = st.sidebar.selectbox("📄 Select Sheet", sheets, key="sheet_selector")
            st.session_state.selected_sheet = selected
            return pd.read_excel(xl, sheet_name=selected, header=None)


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
]
SOON_MODULES = ["🏥  Healthcare", "🏭  Manufacturing", "📣  Marketing", "🎓  Education"]

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
</div>
""", unsafe_allow_html=True)

# ── FILE UPLOAD ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx","csv"], label_visibility="collapsed")

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
if uploaded_file is None:
    st.markdown("""
    <div class='upload-hint'>
        <div style='font-size:2.5rem'>⚡</div>
        <h3>Upload your data to get started</h3>
        <p>Excel or CSV · Any business data · Results in under 60 seconds</p>
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
            sc = st.selectbox("Category", cats)
            if sc != 'All': df_filtered = df_filtered[df_filtered[cat_col] == sc]
        if ware_col:
            wares = ['All'] + sorted(df_inv[ware_col].dropna().unique().tolist())
            sw = st.selectbox("Warehouse", wares)
            if sw != 'All': df_filtered = df_filtered[df_filtered[ware_col] == sw]
        if 'Stock Status' in df_inv.columns:
            statuses = ['All'] + sorted(df_inv['Stock Status'].dropna().unique().tolist())
            ss = st.selectbox("Status", statuses)
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
            sd = st.selectbox("Department", depts)
            if sd != 'All Departments': df_hf = df_hf[df_hf[dept_col] == sd]
        if gender_col:
            genders = ['All'] + sorted(df_hr[gender_col].dropna().unique().tolist())
            sg = st.selectbox("Gender", genders)
            if sg != 'All': df_hf = df_hf[df_hf[gender_col] == sg]
        if status_col:
            statuses = ['All'] + sorted(df_hr[status_col].dropna().unique().tolist())
            ss = st.selectbox("Status", statuses)
            if ss != 'All': df_hf = df_hf[df_hf[status_col] == ss]
        if loc_col:
            locs = ['All'] + sorted(df_hr[loc_col].dropna().unique().tolist())
            sl = st.selectbox("Location", locs)
            if sl != 'All': df_hf = df_hf[df_hf[loc_col] == sl]
        st.caption(f"**{len(df_hf)}** of **{len(df_hr)}** employees")

    # KPIs
    total_emp  = len(df_hf)
    total_pay  = df_hf[salary_col].sum() if salary_col else 0
    avg_salary = df_hf[salary_col].mean() if salary_col else 0
    num_depts  = df_hf[dept_col].nunique() if dept_col else 0
    avg_perf   = df_hf[perf_col].mean() if perf_col else 0
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
                ax.axvline(df_hf[age_col].mean(), color='#ef4444', linestyle='--', label=f"Avg: {df_hf[age_col].mean():.0f} yrs")
                ax.set_title("Age Distribution", fontweight='bold')
                ax.set_xlabel("Age")
                ax.legend()
                plt.tight_layout(); st.pyplot(fig); plt.close()
        with col_b:
            if exp_col:
                fig, ax = plt.subplots(figsize=(6,4))
                clean_chart(fig, ax)
                ax.hist(df_hf[exp_col].dropna(), bins=10, color='#818cf8', edgecolor='white', linewidth=1.5)
                ax.axvline(df_hf[exp_col].mean(), color='#ef4444', linestyle='--', label=f"Avg: {df_hf[exp_col].mean():.1f} yrs")
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
            st_ = st.selectbox("Type", types)
            if st_ != 'All': df_ff = df_ff[df_ff[type_col] == st_]
        if dept_col:
            depts = ['All'] + sorted(df_fin[dept_col].dropna().unique().tolist())
            sd = st.selectbox("Department", depts)
            if sd != 'All': df_ff = df_ff[df_ff[dept_col] == sd]
        st.caption(f"**{len(df_ff)}** of **{len(df_fin)}** records")

    # Separate Revenue vs Expense
    if type_col and amount_col:
        rev_keywords  = ['revenue','income','sales','receipt','inflow']
        exp_keywords  = ['expense','cost','expenditure','payment','outflow','salary','rent']
        rev_mask = df_ff[type_col].str.lower().str.contains('|'.join(rev_keywords), na=False)
        exp_mask = df_ff[type_col].str.lower().str.contains('|'.join(exp_keywords), na=False)
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
        budget_var = df_ff[amount_col].sum() - df_ff[budget_col].sum()

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
            total_budget  = df_ff[budget_col].sum()
            total_actual  = df_ff[amount_col].sum()
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
    total_rev    = df_ret[rev_col].sum() if rev_col else 0
    total_orders = len(df_ret)
    aov          = total_rev / total_orders if total_orders > 0 else 0
    return_rate  = (df_ret[return_col].str.upper() == 'YES').mean() * 100 if return_col else 0
    avg_rating   = df_ret[rating_col].mean() if rating_col else 0
    new_cust_pct = (df_ret[ctype_col].str.title() == 'New').mean() * 100 if ctype_col else 0

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
                ret_counts = df_ret[return_col].str.title().value_counts()
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
                ax.axvline(df_ret[deliv_col].mean(), color='#ef4444', linestyle='--', label=f"Avg: {df_ret[deliv_col].mean():.1f} days")
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
    ontime_rate  = (df_log[ontime_col].str.upper() == 'YES').mean() * 100 if ontime_col else 0
    total_cost   = df_log[cost_col].sum() if cost_col else 0
    damage_rate  = (df_log[damage_col].str.upper() == 'YES').mean() * 100 if damage_col else 0
    avg_rating   = df_log[rating_col].mean() if rating_col else 0
    cost_per_km  = (df_log[cost_col].sum() / df_log[dist_col].sum()) if cost_col and dist_col and df_log[dist_col].sum() > 0 else 0

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
                ot_counts = df_log[ontime_col].str.title().value_counts()
                fig, ax = plt.subplots(figsize=(5,5))
                clean_chart(fig, ax)
                ax.pie(ot_counts.values, labels=ot_counts.index, autopct='%1.1f%%',
                       colors=['#10b981','#ef4444'], wedgeprops=dict(edgecolor='white', linewidth=3), startangle=90)
                ax.set_title(f"On-Time Rate: {ontime_rate:.1f}%", fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()
        with col_b:
            if route_col and ontime_col:
                route_ot = df_log.groupby(route_col).apply(lambda x: (x[ontime_col].str.upper()=='YES').mean()*100).sort_values()
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
                    veh_ot = df_log.groupby(vehicle_col).apply(lambda x: (x[ontime_col].str.upper()=='YES').mean()*100).reset_index()
                    veh_ot.columns = [vehicle_col,'On_Time_%']
                    veh_stats = veh_stats.merge(veh_ot, on=vehicle_col)
                st.dataframe(veh_stats.sort_values('Total_Cost', ascending=False).reset_index(drop=True), use_container_width=True)
        with col_b:
            if driver_col and ontime_col:
                sh("👤","Driver Performance")
                drv_stats = df_log.groupby(driver_col).apply(lambda x: (x[ontime_col].str.upper()=='YES').mean()*100).sort_values(ascending=False).reset_index()
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
                cargo_dmg = df_log.groupby(cargo_col).apply(lambda x: (x[damage_col].str.upper()=='YES').mean()*100).reset_index()
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

    total_rev     = df_rest[rev_col].sum() if rev_col else 0
    total_orders  = len(df_rest)
    avg_rating    = df_rest[rating_col].mean() if rating_col else 0
    occupancy_rate= (df_rest[tables_col] / df_rest[total_t_col]).mean() * 100 if tables_col and total_t_col else 0
    if sell_col and cost_col:
        df_rest['Food Cost %'] = (df_rest[cost_col] / df_rest[sell_col] * 100).round(1)
        avg_food_cost = df_rest['Food Cost %'].mean()
    else:
        avg_food_cost = 0
    total_waste = df_rest[waste_col].sum() if waste_col else 0

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
        sr = st.selectbox("Region", regions)
        if sr != 'All': df_s = df_s[df_s[region_col] == sr]
    if cat_col:
        cats = ['All'] + sorted(df_clean[cat_col].dropna().unique().tolist())
        sc = st.selectbox("Category", cats)
        if sc != 'All': df_s = df_s[df_s[cat_col] == sc]
    if person_col:
        persons = ['All'] + sorted(df_clean[person_col].dropna().unique().tolist())
        sp = st.selectbox("Salesperson", persons)
        if sp != 'All': df_s = df_s[df_s[person_col] == sp]
    st.caption(f"**{len(df_s):,}** of **{len(df_clean):,}** rows")

# KPIs
total_rev    = df_s[rev_col].sum() if rev_col else 0
total_qty    = df_s[qty_col].sum() if qty_col else 0
total_target = df_s[target_col].sum() if target_col else 0
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
        if rev_col: metrics.append(["Total Revenue", f"₹{df_s[rev_col].sum():,.0f}"])
        if target_col: metrics.append(["Target Achievement", f"{achievement:.1f}%"])
        if qty_col: metrics.append(["Total Units Sold", f"{df_s[qty_col].sum():,.0f}"])
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
