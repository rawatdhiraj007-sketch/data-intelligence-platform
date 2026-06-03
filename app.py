import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.units import inch
import io
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Velytics — Your data scientist. No hiring required.", page_icon="⚡", layout="wide")

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #f9fafb; }

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border-left: 4px solid #6366f1;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin: 8px 0;
}
.section-header {
    background: #0a0a0a;
    color: white;
    padding: 12px 20px;
    border-radius: 10px;
    margin: 28px 0 16px 0;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: -0.3px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.alert-box {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 6px 0;
    font-size: 0.9rem;
}
.insight-box {
    background: #f0fdf4;
    border-left: 4px solid #10b981;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 6px 0;
    font-size: 0.9rem;
}
.hero-banner {
    background: #0a0a0a;
    border-radius: 16px;
    padding: 40px;
    margin-bottom: 24px;
    text-align: center;
}
.hero-banner h1 {
    color: white;
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: -1.5px;
    margin: 0 0 8px 0;
    line-height: 1.1;
}
.hero-banner h1 span { color: #818cf8; }
.hero-banner p {
    color: #9ca3af;
    font-size: 1.05rem;
    margin: 0;
}
.upload-area {
    background: white;
    border: 2px dashed #e5e7eb;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
    transition: border-color 0.2s;
}
.feature-pill {
    display: inline-block;
    background: #eef2ff;
    color: #4f46e5;
    border-radius: 100px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-banner'>
    <h1>Vely<span>tics</span></h1>
    <p>Your data scientist. No hiring required.</p>
</div>
""", unsafe_allow_html=True)

# ── Module Selector ──────────────────────────────────────────────────────────
col_m1, col_m2, col_m3 = st.columns([1,1,4])
with col_m1:
    module = st.selectbox("Select Module", ["Sales Intelligence", "Inventory Intelligence"], label_visibility="collapsed")
with col_m2:
    st.markdown("<div style='padding-top:8px; color:#6b7280; font-size:0.85rem;'>More modules coming soon</div>", unsafe_allow_html=True)

st.markdown("""
<div style='margin-bottom:8px; margin-top:4px;'>
    <span class='feature-pill'>⚡ Auto-cleaning</span>
    <span class='feature-pill'>📈 Forecasting</span>
    <span class='feature-pill'>🚨 Smart alerts</span>
    <span class='feature-pill'>📄 PDF report</span>
    <span class='feature-pill'>🔒 Data never stored</span>
</div>
""", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload your Excel or CSV file to get started", type=["xlsx", "csv"])

if uploaded_file is not None:

    # Load
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # ════════════════════════════════════════════════════════════════════════
    # INVENTORY INTELLIGENCE MODULE
    # ════════════════════════════════════════════════════════════════════════
    if module == "Inventory Intelligence":

        # ── Auto Clean ───────────────────────────────────────────────────────
        df_inv = df.copy()
        issues_fixed_inv = 0
        for col in df_inv.select_dtypes(include='number').columns:
            missing = df_inv[col].isnull().sum()
            if missing > 0:
                df_inv[col].fillna(df_inv[col].median(), inplace=True)
                issues_fixed_inv += missing
        df_inv.drop_duplicates(inplace=True)

        # ── Detect Columns ───────────────────────────────────────────────────
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

        # ── Stock Status Classification ──────────────────────────────────────
        if stock_col and reorder_col:
            df_inv['Stock Status'] = 'Normal'
            df_inv.loc[df_inv[stock_col] == 0, 'Stock Status'] = 'Out of Stock'
            df_inv.loc[(df_inv[stock_col] > 0) & (df_inv[stock_col] <= df_inv[reorder_col]), 'Stock Status'] = 'Low Stock'
            if max_col and max_col in df_inv.columns:
                df_inv.loc[df_inv[stock_col] >= df_inv[max_col] * 0.9, 'Stock Status'] = 'Overstocked'

        # ── Days of Stock Left ───────────────────────────────────────────────
        if stock_col and demand_col:
            df_inv['Days of Stock Left'] = (df_inv[stock_col] / (df_inv[demand_col] / 30)).round(1)
            df_inv['Days of Stock Left'] = df_inv['Days of Stock Left'].replace([float('inf'), -float('inf')], 999)

        # ── Margin ───────────────────────────────────────────────────────────
        if cost_col and price_col:
            df_inv['Margin ($)'] = df_inv[price_col] - df_inv[cost_col]
            df_inv['Margin %'] = ((df_inv['Margin ($)'] / df_inv[price_col]) * 100).round(1)

        # ── Stock Value ──────────────────────────────────────────────────────
        if stock_col and cost_col:
            df_inv['Stock Value ($)'] = (df_inv[stock_col] * df_inv[cost_col]).round(0)

        # ── Dead Stock ───────────────────────────────────────────────────────
        dead_stock = pd.DataFrame()
        if last_sold_col and stock_col:
            dead_stock = df_inv[(df_inv[last_sold_col] > 90) & (df_inv[stock_col] > 0)].copy()

        st.success(f"✅ Inventory file loaded — {len(df_inv):,} products · {len(df_inv.columns)} columns · {issues_fixed_inv} issues auto-fixed")

        # ── SIDEBAR FILTERS ──────────────────────────────────────────────────
        with st.sidebar:
            st.markdown("### 🔍 Filter Inventory")
            df_filtered = df_inv.copy()

            if cat_col:
                all_cats = ['All Categories'] + sorted(df_inv[cat_col].dropna().unique().tolist())
                sel_cat = st.selectbox("Category", all_cats)
                if sel_cat != 'All Categories':
                    df_filtered = df_filtered[df_filtered[cat_col] == sel_cat]

            if ware_col:
                all_ware = ['All Warehouses'] + sorted(df_inv[ware_col].dropna().unique().tolist())
                sel_ware = st.selectbox("Warehouse", all_ware)
                if sel_ware != 'All Warehouses':
                    df_filtered = df_filtered[df_filtered[ware_col] == sel_ware]

            if 'Stock Status' in df_inv.columns:
                all_status = ['All Statuses'] + sorted(df_inv['Stock Status'].dropna().unique().tolist())
                sel_status = st.selectbox("Stock Status", all_status)
                if sel_status != 'All Statuses':
                    df_filtered = df_filtered[df_filtered['Stock Status'] == sel_status]

            if supplier_col:
                all_sup = ['All Suppliers'] + sorted(df_inv[supplier_col].dropna().unique().tolist())
                sel_sup = st.selectbox("Supplier", all_sup)
                if sel_sup != 'All Suppliers':
                    df_filtered = df_filtered[df_filtered[supplier_col] == sel_sup]

            st.caption(f"Showing **{len(df_filtered)}** of **{len(df_inv)}** products")

        # ── KPI CARDS ────────────────────────────────────────────────────────
        inv_val     = df_filtered['Stock Value ($)'].sum() if 'Stock Value ($)' in df_filtered.columns else 0
        total_skus  = df_filtered[prod_col].nunique() if prod_col else len(df_filtered)
        oos_count   = (df_filtered[stock_col] == 0).sum() if stock_col else 0
        low_count   = (df_filtered['Stock Status'] == 'Low Stock').sum() if 'Stock Status' in df_filtered.columns else 0
        over_count  = (df_filtered['Stock Status'] == 'Overstocked').sum() if 'Stock Status' in df_filtered.columns else 0
        dead_count  = len(dead_stock[dead_stock.index.isin(df_filtered.index)]) if len(dead_stock) > 0 else 0

        st.markdown(f"""
        <div style='display:grid; grid-template-columns:repeat(6,1fr); gap:12px; margin:16px 0 24px 0;'>
            <div style='background:white; border-radius:12px; padding:16px 12px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.06); border-top:3px solid #6366f1;'>
                <div style='font-size:1.5rem; font-weight:800; color:#0a0a0a;'>{total_skus}</div>
                <div style='font-size:0.72rem; color:#6b7280; margin-top:4px; font-weight:500;'>TOTAL SKUs</div>
            </div>
            <div style='background:white; border-radius:12px; padding:16px 12px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.06); border-top:3px solid #6366f1;'>
                <div style='font-size:1.5rem; font-weight:800; color:#0a0a0a;'>${inv_val:,.0f}</div>
                <div style='font-size:0.72rem; color:#6b7280; margin-top:4px; font-weight:500;'>INVENTORY VALUE</div>
            </div>
            <div style='background:white; border-radius:12px; padding:16px 12px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.06); border-top:3px solid #ef4444;'>
                <div style='font-size:1.5rem; font-weight:800; color:#ef4444;'>{oos_count}</div>
                <div style='font-size:0.72rem; color:#6b7280; margin-top:4px; font-weight:500;'>OUT OF STOCK</div>
            </div>
            <div style='background:white; border-radius:12px; padding:16px 12px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.06); border-top:3px solid #f59e0b;'>
                <div style='font-size:1.5rem; font-weight:800; color:#f59e0b;'>{low_count}</div>
                <div style='font-size:0.72rem; color:#6b7280; margin-top:4px; font-weight:500;'>LOW STOCK</div>
            </div>
            <div style='background:white; border-radius:12px; padding:16px 12px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.06); border-top:3px solid #6366f1;'>
                <div style='font-size:1.5rem; font-weight:800; color:#6366f1;'>{over_count}</div>
                <div style='font-size:0.72rem; color:#6b7280; margin-top:4px; font-weight:500;'>OVERSTOCKED</div>
            </div>
            <div style='background:white; border-radius:12px; padding:16px 12px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.06); border-top:3px solid #9ca3af;'>
                <div style='font-size:1.5rem; font-weight:800; color:#9ca3af;'>{dead_count}</div>
                <div style='font-size:0.72rem; color:#6b7280; margin-top:4px; font-weight:500;'>DEAD STOCK</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── TABS ─────────────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🚦 Stock Status", "💀 Dead Stock", "🏆 ABC Analysis",
            "🔔 Reorder Alerts", "📊 Category & Margins", "🚨 Smart Alerts"
        ])

        # ══ TAB 1: STOCK STATUS ══════════════════════════════════════════════
        with tab1:
            if stock_col and reorder_col and 'Stock Status' in df_filtered.columns:
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    status_counts = df_filtered['Stock Status'].value_counts()
                    colors_map = {'Out of Stock': '#ef4444', 'Low Stock': '#f59e0b', 'Normal': '#10b981', 'Overstocked': '#6366f1'}
                    bar_colors = [colors_map.get(s, '#6366f1') for s in status_counts.index]
                    fig, ax = plt.subplots(figsize=(6, 4))
                    fig.patch.set_facecolor('white')
                    bars = ax.bar(status_counts.index, status_counts.values, color=bar_colors, width=0.5, edgecolor='white', linewidth=2)
                    ax.set_ylabel("Products", fontsize=11)
                    ax.set_title("Stock Status Distribution", fontweight='bold', fontsize=13, pad=12)
                    ax.spines[['top','right']].set_visible(False)
                    ax.set_facecolor('#f9fafb')
                    for bar, val in zip(bars, status_counts.values):
                        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, str(val),
                                ha='center', va='bottom', fontweight='bold', fontsize=12)
                    plt.tight_layout()
                    st.pyplot(fig); plt.close()

                with col_b:
                    # Status summary cards
                    for status, color in [('Out of Stock','#ef4444'),('Low Stock','#f59e0b'),('Normal','#10b981'),('Overstocked','#6366f1')]:
                        count = (df_filtered['Stock Status'] == status).sum()
                        pct = count / len(df_filtered) * 100 if len(df_filtered) > 0 else 0
                        st.markdown(f"""
                        <div style='display:flex; justify-content:space-between; align-items:center;
                             background:white; border-radius:10px; padding:12px 16px; margin:6px 0;
                             border-left:4px solid {color}; box-shadow:0 1px 4px rgba(0,0,0,0.06);'>
                            <span style='font-weight:600; color:#0a0a0a;'>{status}</span>
                            <span style='font-size:1.2rem; font-weight:800; color:{color};'>{count} <span style='font-size:0.75rem; color:#6b7280;'>({pct:.0f}%)</span></span>
                        </div>""", unsafe_allow_html=True)

                st.markdown("#### 📋 Products Needing Action")
                urgent = df_filtered[df_filtered['Stock Status'].isin(['Out of Stock', 'Low Stock'])].copy()
                if len(urgent) > 0:
                    show_cols = [c for c in [prod_col, cat_col, ware_col, stock_col, reorder_col, 'Stock Status'] if c and c in urgent.columns]
                    urgent_display = urgent[show_cols].sort_values('Stock Status').reset_index(drop=True)
                    st.dataframe(
                        urgent_display.style.apply(
                            lambda col: ['background-color: #fef2f2; color: #991b1b' if v == 'Out of Stock'
                                         else 'background-color: #fffbeb; color: #92400e' if v == 'Low Stock'
                                         else '' for v in col], subset=['Stock Status']
                        ) if 'Stock Status' in urgent_display.columns else urgent_display,
                        use_container_width=True, height=400
                    )
                else:
                    st.markdown("<div class='insight-box'>✅ All products have healthy stock levels!</div>", unsafe_allow_html=True)

        # ══ TAB 2: DEAD STOCK ════════════════════════════════════════════════
        with tab2:
            dead_filtered = df_filtered[(df_filtered[last_sold_col] > 90) & (df_filtered[stock_col] > 0)].copy() if last_sold_col and stock_col else pd.DataFrame()
            if 'Stock Value ($)' in dead_filtered.columns:
                dead_locked = dead_filtered['Stock Value ($)'].sum()
            else:
                dead_locked = 0

            if len(dead_filtered) > 0:
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Dead Stock Products", len(dead_filtered))
                col_b.metric("Capital Locked", f"${dead_locked:,.0f}")
                col_c.metric("% of Inventory", f"{dead_locked/inv_val*100:.1f}%" if inv_val > 0 else "N/A")

                st.markdown(f"<div class='alert-box'>⚠️ <b>{len(dead_filtered)} products</b> haven't sold in 90+ days. Consider discounting, bundling, or liquidating to free up <b>${dead_locked:,.0f}</b> in capital.</div>", unsafe_allow_html=True)

                show_cols = [c for c in [prod_col, cat_col, ware_col, stock_col, last_sold_col, 'Stock Value ($)'] if c and c in dead_filtered.columns]
                st.dataframe(
                    dead_filtered[show_cols].sort_values('Stock Value ($)', ascending=False).reset_index(drop=True),
                    use_container_width=True, height=400
                )

                if cat_col and 'Stock Value ($)' in dead_filtered.columns:
                    st.markdown("#### Dead Stock by Category")
                    dead_cat = dead_filtered.groupby(cat_col)['Stock Value ($)'].sum().sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(8, 3))
                    fig.patch.set_facecolor('white')
                    ax.barh(dead_cat.index, dead_cat.values, color='#ef4444', alpha=0.8)
                    ax.set_xlabel("Capital Locked ($)")
                    ax.set_facecolor('#f9fafb')
                    ax.spines[['top','right']].set_visible(False)
                    for i, val in enumerate(dead_cat.values):
                        ax.text(val, i, f'  ${val:,.0f}', va='center', fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig); plt.close()
            else:
                st.markdown("<div class='insight-box'>✅ No dead stock detected — all inventory is moving!</div>", unsafe_allow_html=True)

        # ══ TAB 3: ABC ANALYSIS ══════════════════════════════════════════════
        with tab3:
            if prod_col and sold90_col:
                st.caption("**A** = Top products driving 80% of sales (protect & prioritize) · **B** = Middle tier · **C** = Slow movers (review)")

                abc = df_filtered.groupby(prod_col)[sold90_col].sum().sort_values(ascending=False).reset_index()
                abc.columns = ['Product', 'Units Sold (90d)']
                total_sold = abc['Units Sold (90d)'].sum()
                abc['Cumulative %'] = (abc['Units Sold (90d)'].cumsum() / total_sold * 100).round(1) if total_sold > 0 else 0
                abc['ABC Class'] = 'C'
                abc.loc[abc['Cumulative %'] <= 80, 'ABC Class'] = 'A'
                abc.loc[(abc['Cumulative %'] > 80) & (abc['Cumulative %'] <= 95), 'ABC Class'] = 'B'

                col_a, col_b = st.columns([1, 1])
                with col_a:
                    abc_counts = abc['ABC Class'].value_counts().reindex(['A','B','C'], fill_value=0)
                    abc_colors_list = ['#10b981','#6366f1','#9ca3af']
                    fig, ax = plt.subplots(figsize=(5, 5))
                    fig.patch.set_facecolor('white')
                    wedges, texts, autotexts = ax.pie(
                        abc_counts.values, labels=[f'Class {c}' for c in abc_counts.index],
                        autopct='%1.0f%%', colors=abc_colors_list, startangle=90,
                        wedgeprops=dict(edgecolor='white', linewidth=2)
                    )
                    for t in autotexts: t.set_fontweight('bold')
                    ax.set_title("ABC Class Distribution", fontweight='bold', fontsize=13)
                    plt.tight_layout()
                    st.pyplot(fig); plt.close()

                with col_b:
                    for cls, color, label in [('A','#10b981','Star Products — Never let these go OOS'),
                                               ('B','#6366f1','Growing Products — Monitor closely'),
                                               ('C','#9ca3af','Slow Movers — Review or discount')]:
                        cls_data = abc[abc['ABC Class'] == cls]
                        cls_units = cls_data['Units Sold (90d)'].sum()
                        st.markdown(f"""
                        <div style='background:white; border-radius:10px; padding:14px 16px; margin:8px 0;
                             border-left:4px solid {color}; box-shadow:0 1px 4px rgba(0,0,0,0.06);'>
                            <div style='font-weight:800; font-size:1rem; color:{color};'>Class {cls} — {len(cls_data)} products</div>
                            <div style='font-size:0.82rem; color:#6b7280; margin-top:2px;'>{label}</div>
                            <div style='font-size:0.82rem; color:#0a0a0a; margin-top:4px; font-weight:600;'>{cls_units:,} units sold in 90 days</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("#### Full ABC Table")
                abc_styled = abc.copy()
                st.dataframe(
                    abc_styled.style.apply(
                        lambda col: ['background-color:#f0fdf4; color:#166534' if v=='A'
                                     else 'background-color:#eef2ff; color:#3730a3' if v=='B'
                                     else 'background-color:#f9fafb; color:#4b5563' for v in col],
                        subset=['ABC Class']
                    ),
                    use_container_width=True, height=400
                )

        # ══ TAB 4: REORDER ALERTS ════════════════════════════════════════════
        with tab4:
            if stock_col and reorder_col and demand_col and prod_col and 'Days of Stock Left' in df_filtered.columns:
                reorder_df = df_filtered[df_filtered['Days of Stock Left'] < 30].sort_values('Days of Stock Left').copy()
                critical_df = df_filtered[df_filtered['Days of Stock Left'] < 7].copy()

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Reorder Needed (<30 days)", len(reorder_df), delta="Act now" if len(reorder_df)>0 else None, delta_color="inverse")
                col_b.metric("Critical (<7 days)", len(critical_df), delta="URGENT" if len(critical_df)>0 else None, delta_color="inverse")
                col_c.metric("Healthy Stock (>30 days)", len(df_filtered) - len(reorder_df))

                if len(reorder_df) > 0:
                    st.markdown(f"<div class='alert-box'>⚠️ <b>{len(reorder_df)} products</b> will run out within 30 days. Place orders immediately!</div>", unsafe_allow_html=True)

                    # Bar chart of days left
                    top_urgent = reorder_df.head(15)
                    if prod_col in top_urgent.columns:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        fig.patch.set_facecolor('white')
                        bar_c = ['#ef4444' if d < 7 else '#f59e0b' for d in top_urgent['Days of Stock Left']]
                        ax.barh(top_urgent[prod_col].str[:25], top_urgent['Days of Stock Left'], color=bar_c)
                        ax.axvline(x=7, color='#ef4444', linestyle='--', alpha=0.6, label='Critical (7 days)')
                        ax.axvline(x=14, color='#f59e0b', linestyle='--', alpha=0.6, label='Warning (14 days)')
                        ax.set_xlabel("Days of Stock Remaining")
                        ax.set_title("Products Running Low — Days of Stock Left", fontweight='bold')
                        ax.set_facecolor('#f9fafb')
                        ax.spines[['top','right']].set_visible(False)
                        ax.legend(fontsize=9)
                        plt.tight_layout()
                        st.pyplot(fig); plt.close()

                    show_cols = [c for c in [prod_col, cat_col, ware_col, stock_col, demand_col, 'Days of Stock Left', supplier_col] if c and c in reorder_df.columns]
                    st.dataframe(
                        reorder_df[show_cols].reset_index(drop=True).style.apply(
                            lambda col: ['background-color:#fef2f2' if v < 7
                                         else 'background-color:#fffbeb' if v < 14
                                         else '' for v in col], subset=['Days of Stock Left']
                        ),
                        use_container_width=True, height=350
                    )
                else:
                    st.markdown("<div class='insight-box'>✅ All products have 30+ days of stock. No immediate reorders needed.</div>", unsafe_allow_html=True)

        # ══ TAB 5: CATEGORY & MARGINS ════════════════════════════════════════
        with tab5:
            col_a, col_b = st.columns(2)

            with col_a:
                if cat_col and 'Stock Value ($)' in df_filtered.columns:
                    st.markdown("#### Stock Value by Category")
                    cat_val = df_filtered.groupby(cat_col)['Stock Value ($)'].sum().sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    fig.patch.set_facecolor('white')
                    colors_grad = ['#6366f1','#818cf8','#a5b4fc','#c7d2fe','#e0e7ff','#4f46e5']
                    ax.barh(cat_val.index, cat_val.values, color=colors_grad[:len(cat_val)])
                    ax.set_xlabel("Value ($)")
                    ax.invert_yaxis()
                    ax.set_facecolor('#f9fafb')
                    ax.spines[['top','right']].set_visible(False)
                    for i, val in enumerate(cat_val.values):
                        ax.text(val, i, f'  ${val:,.0f}', va='center', fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig); plt.close()

            with col_b:
                if cat_col and sold90_col:
                    st.markdown("#### Units Sold by Category (90 Days)")
                    cat_sold = df_filtered.groupby(cat_col)[sold90_col].sum().sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    fig.patch.set_facecolor('white')
                    wedges, texts, autotexts = ax.pie(
                        cat_sold.values, labels=cat_sold.index, autopct='%1.1f%%',
                        colors=['#6366f1','#818cf8','#a5b4fc','#c7d2fe','#e0e7ff','#4f46e5'],
                        wedgeprops=dict(edgecolor='white', linewidth=2), startangle=90
                    )
                    for t in autotexts: t.set_fontweight('bold')
                    plt.tight_layout()
                    st.pyplot(fig); plt.close()

            if cost_col and price_col and prod_col and 'Margin %' in df_filtered.columns:
                st.markdown("#### Profit Margin Analysis")
                col_c, col_d = st.columns(2)

                with col_c:
                    st.markdown("**🏆 Top 10 Highest Margin**")
                    top_margin = df_filtered[[prod_col, cost_col, price_col, 'Margin %']].sort_values('Margin %', ascending=False).head(10).reset_index(drop=True)
                    st.dataframe(top_margin.style.background_gradient(subset=['Margin %'], cmap='Greens'), use_container_width=True)

                with col_d:
                    st.markdown("**⚠️ Bottom 10 Lowest Margin**")
                    low_margin = df_filtered[[prod_col, cost_col, price_col, 'Margin %']].sort_values('Margin %').head(10).reset_index(drop=True)
                    st.dataframe(low_margin.style.background_gradient(subset=['Margin %'], cmap='Reds_r'), use_container_width=True)

        # ══ TAB 6: SMART ALERTS ══════════════════════════════════════════════
        with tab6:
            inv_alerts = []
            if stock_col:
                oos = (df_filtered[stock_col] == 0).sum()
                if oos > 0:
                    inv_alerts.append(('🔴', 'critical', f"<b>{oos} products</b> are completely OUT OF STOCK — immediate action required"))
            if 'Days of Stock Left' in df_filtered.columns:
                crit7 = (df_filtered['Days of Stock Left'] < 7).sum()
                if crit7 > 0:
                    inv_alerts.append(('🔴', 'critical', f"<b>{crit7} products</b> will run out within 7 days — URGENT reorder needed"))
            if len(dead_filtered) > 0:
                inv_alerts.append(('⚠️', 'warning', f"<b>{len(dead_filtered)} products</b> are dead stock — consider discounting or liquidating (${dead_locked:,.0f} locked)"))
            if 'Stock Status' in df_filtered.columns:
                over = (df_filtered['Stock Status'] == 'Overstocked').sum()
                if over > 0:
                    inv_stock_val = df_filtered[df_filtered['Stock Status']=='Overstocked']['Stock Value ($)'].sum() if 'Stock Value ($)' in df_filtered.columns else 0
                    inv_alerts.append(('⚠️', 'warning', f"<b>{over} products</b> are overstocked — ${inv_stock_val:,.0f} tied up unnecessarily"))
            if 'Margin %' in df_filtered.columns:
                neg_margin = (df_filtered['Margin %'] < 0).sum()
                if neg_margin > 0:
                    inv_alerts.append(('🔴', 'critical', f"<b>{neg_margin} products</b> have NEGATIVE margins — selling below cost!"))
                low_margin_count = ((df_filtered['Margin %'] >= 0) & (df_filtered['Margin %'] < 10)).sum()
                if low_margin_count > 0:
                    inv_alerts.append(('⚠️', 'warning', f"<b>{low_margin_count} products</b> have margins below 10% — consider price review"))

            if len(inv_alerts) == 0:
                st.markdown("<div class='insight-box' style='padding:20px; font-size:1rem;'>✅ All clear! Inventory looks healthy — no critical alerts at this time.</div>", unsafe_allow_html=True)
            else:
                critical_alerts = [(e,m) for e,t,m in inv_alerts if t=='critical']
                warning_alerts  = [(e,m) for e,t,m in inv_alerts if t=='warning']

                if critical_alerts:
                    st.markdown("#### 🔴 Critical — Act Now")
                    for e, msg in critical_alerts:
                        st.markdown(f"""<div style='background:#fef2f2; border-left:4px solid #ef4444;
                            padding:14px 16px; border-radius:8px; margin:6px 0; font-size:0.9rem;'>
                            {e} {msg}</div>""", unsafe_allow_html=True)

                if warning_alerts:
                    st.markdown("#### ⚠️ Warnings — Monitor")
                    for e, msg in warning_alerts:
                        st.markdown(f"""<div style='background:#fffbeb; border-left:4px solid #f59e0b;
                            padding:14px 16px; border-radius:8px; margin:6px 0; font-size:0.9rem;'>
                            {e} {msg}</div>""", unsafe_allow_html=True)

        st.stop()

    # ── Auto Clean ───────────────────────────────────────────────────────────
    df_clean = df.copy()
    issues_fixed = 0
    for col in df_clean.select_dtypes(include='number').columns:
        missing = df_clean[col].isnull().sum()
        if missing > 0:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
            issues_fixed += missing
    df_clean.drop_duplicates(inplace=True)

    # ── Detect Columns ───────────────────────────────────────────────────────
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

    st.success(f"✅ File loaded & cleaned — {len(df_clean):,} rows · {len(df_clean.columns)} columns · {issues_fixed} issues auto-fixed · Ready for analysis")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1: KEY METRICS
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-header'>📈 Key Business Metrics</div>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    if rev_col:
        total_rev = df_clean[rev_col].sum()
        c1.metric("Total Revenue", f"₹{total_rev:,.0f}")
        if target_col:
            total_target = df_clean[target_col].sum()
            achievement = (total_rev / total_target * 100)
            c2.metric("Target Achievement", f"{achievement:.1f}%", f"Target: ₹{total_target:,.0f}")
    if qty_col:
        c3.metric("Total Units Sold", f"{df_clean[qty_col].sum():,.0f}")
    if product_col:
        c4.metric("Products", f"{df_clean[product_col].nunique()}")
    if person_col:
        c5.metric("Salespersons", f"{df_clean[person_col].nunique()}")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 2: REVENUE ANALYSIS
    # ════════════════════════════════════════════════════════════════════════
    if rev_col:
        st.markdown("<div class='section-header'>🌍 Revenue Analysis</div>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        # Revenue by Region
        if region_col:
            with col_a:
                st.write("**Revenue by Region**")
                region_rev = df_clean.groupby(region_col)[rev_col].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.bar(region_rev.index, region_rev.values,
                              color=['#6366f1','#818cf8','#a5b4fc','#c7d2fe'])
                ax.set_ylabel("Revenue (₹)")
                for bar, val in zip(bars, region_rev.values):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                            f'₹{val/100000:.1f}L', ha='center', va='bottom', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        # Revenue by Category
        if cat_col:
            with col_b:
                st.write("**Revenue by Category**")
                cat_rev = df_clean.groupby(cat_col)[rev_col].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.pie(cat_rev.values, labels=cat_rev.index, autopct='%1.1f%%',
                       colors=['#6366f1','#818cf8','#a5b4fc','#c7d2fe','#e0e7ff'])
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 3: PRODUCT / SKU INTELLIGENCE
    # ════════════════════════════════════════════════════════════════════════
    if product_col and rev_col:
        st.markdown("<div class='section-header'>📦 Product / SKU Intelligence</div>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.write("**Top 10 Products by Revenue**")
            top_products = df_clean.groupby(product_col)[rev_col].sum().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(6, 5))
            top_products.plot(kind='barh', ax=ax, color='#6366f1')
            ax.set_xlabel("Revenue (₹)")
            ax.invert_yaxis()
            for i, val in enumerate(top_products.values):
                ax.text(val, i, f' ₹{val/100000:.1f}L', va='center', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_b:
            if qty_col:
                st.write("**Top 10 Products by Units Sold**")
                top_qty = df_clean.groupby(product_col)[qty_col].sum().sort_values(ascending=False).head(10)
                fig, ax = plt.subplots(figsize=(6, 5))
                top_qty.plot(kind='barh', ax=ax, color='#6366f1')
                ax.set_xlabel("Units Sold")
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        # Slow moving products
        if qty_col:
            st.write("**⚠️ Slow Moving Products (Bottom 5)**")
            slow = df_clean.groupby(product_col)[qty_col].sum().sort_values().head(5).reset_index()
            slow.columns = ['Product', 'Units Sold']
            st.table(slow)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 4: PRICE INCONSISTENCY DETECTION
    # ════════════════════════════════════════════════════════════════════════
    if product_col and price_col:
        st.markdown("<div class='section-header'>💰 Price Inconsistency Detection</div>", unsafe_allow_html=True)

        price_analysis = df_clean.groupby(product_col)[price_col].agg(['min','max','mean','std']).reset_index()
        price_analysis.columns = ['Product', 'Min Price (₹)', 'Max Price (₹)', 'Avg Price (₹)', 'Std Dev']
        price_analysis['Price Variation (₹)'] = price_analysis['Max Price (₹)'] - price_analysis['Min Price (₹)']
        price_analysis['Inconsistent'] = price_analysis['Price Variation (₹)'] > 0

        inconsistent = price_analysis[price_analysis['Inconsistent']].sort_values('Price Variation (₹)', ascending=False)

        if len(inconsistent) > 0:
            st.markdown(f"<div class='alert-box'>⚠️ <b>{len(inconsistent)} products</b> found with inconsistent pricing — same product billed at different prices!</div>", unsafe_allow_html=True)
            display_cols = ['Product', 'Min Price (₹)', 'Max Price (₹)', 'Price Variation (₹)']
            st.dataframe(inconsistent[display_cols].round(0).reset_index(drop=True), use_container_width=True)
        else:
            st.markdown("<div class='insight-box'>✅ No price inconsistencies found — pricing is consistent.</div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 5: SALESPERSON PERFORMANCE
    # ════════════════════════════════════════════════════════════════════════
    if person_col and rev_col:
        st.markdown("<div class='section-header'>👤 Salesperson Performance</div>", unsafe_allow_html=True)

        person_data = df_clean.groupby(person_col).agg(
            Revenue=(rev_col, 'sum'),
            Transactions=(rev_col, 'count')
        ).reset_index()

        if target_col:
            person_target = df_clean.groupby(person_col)[target_col].sum().reset_index()
            person_target.columns = [person_col, 'Target']
            person_data = person_data.merge(person_target, on=person_col)
            person_data['Achievement %'] = (person_data['Revenue'] / person_data['Target'] * 100).round(1)

        person_data = person_data.sort_values('Revenue', ascending=False).reset_index(drop=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Revenue by Salesperson**")
            fig, ax = plt.subplots(figsize=(6, 5))
            bars = ax.bar(person_data[person_col], person_data['Revenue'], color='#6366f1')
            ax.set_ylabel("Revenue (₹)")
            plt.xticks(rotation=45, ha='right', fontsize=8)
            for bar, val in zip(bars, person_data['Revenue']):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                        f'₹{val/100000:.1f}L', ha='center', va='bottom', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.write("**Performance Table**")
            display_df = person_data.copy()
            display_df['Revenue'] = display_df['Revenue'].apply(lambda x: f"₹{x:,.0f}")
            if 'Achievement %' in display_df.columns:
                display_df['Target'] = display_df['Target'].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(display_df, use_container_width=True)

        # Top and bottom performers
        top = person_data.iloc[0][person_col]
        bottom = person_data.iloc[-1][person_col]
        st.markdown(f"<div class='insight-box'>🏆 <b>Top Performer:</b> {top} | ⚠️ <b>Needs Attention:</b> {bottom}</div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 6: CUSTOMER INTELLIGENCE
    # ════════════════════════════════════════════════════════════════════════
    if ctype_col and rev_col:
        st.markdown("<div class='section-header'>👥 Customer Intelligence</div>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.write("**Revenue by Customer Type**")
            ctype_rev = df_clean.groupby(ctype_col)[rev_col].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(ctype_rev.values, labels=ctype_rev.index, autopct='%1.1f%%',
                   colors=['#6366f1','#818cf8','#a5b4fc','#c7d2fe'])
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.write("**Customer Type Summary**")
            ctype_summary = df_clean.groupby(ctype_col).agg(
                Revenue=(rev_col, 'sum'),
                Orders=(rev_col, 'count'),
            ).reset_index()
            ctype_summary['Avg Order (₹)'] = (ctype_summary['Revenue'] / ctype_summary['Orders']).round(0)
            ctype_summary['Revenue'] = ctype_summary['Revenue'].apply(lambda x: f"₹{x:,.0f}")
            ctype_summary['Avg Order (₹)'] = ctype_summary['Avg Order (₹)'].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(ctype_summary, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 7: PAYMENT INTELLIGENCE
    # ════════════════════════════════════════════════════════════════════════
    if payment_col and rev_col:
        st.markdown("<div class='section-header'>💳 Payment Intelligence</div>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Revenue by Payment Mode**")
            pay_rev = df_clean.groupby(payment_col)[rev_col].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(pay_rev.index, pay_rev.values, color=['#1F4E79','#2E75B6','#9DC3E6','#BDD7EE','#DEEAF1'])
            ax.set_ylabel("Revenue (₹)")
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.write("**Payment Mode Breakdown**")
            pay_summary = df_clean.groupby(payment_col).agg(
                Revenue=(rev_col, 'sum'),
                Transactions=(rev_col, 'count')
            ).reset_index()
            pay_summary['Revenue %'] = (pay_summary['Revenue'] / pay_summary['Revenue'].sum() * 100).round(1)
            pay_summary['Revenue'] = pay_summary['Revenue'].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(pay_summary, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 8: SALES FORECAST
    # ════════════════════════════════════════════════════════════════════════
    forecast_values = None
    future_months = None
    growth_pct = None
    forecast_buf = None

    if date_col and rev_col:
        st.markdown("<div class='section-header'>🔮 Sales Forecast — Next 6 Months</div>", unsafe_allow_html=True)

        monthly = df_clean.groupby(df_clean[date_col].dt.to_period('M'))[rev_col].sum().reset_index()
        monthly.columns = ['Month', 'Revenue']
        monthly['MonthNum'] = range(len(monthly))

        X = monthly['MonthNum'].values.reshape(-1, 1)
        y = monthly['Revenue'].values
        model = LinearRegression()
        model.fit(X, y)

        future_months = pd.period_range(start=monthly['Month'].iloc[-1] + 1, periods=6, freq='M')
        future_nums = np.array(range(len(monthly), len(monthly) + 6)).reshape(-1, 1)
        forecast_values = np.maximum(model.predict(future_nums), 0)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(monthly['MonthNum'], monthly['Revenue'], 'b-o', label='Historical Revenue', linewidth=2)
        ax.plot(future_nums, forecast_values, 'r--o', label='Forecasted Revenue', linewidth=2)
        ax.axvline(x=len(monthly)-0.5, color='gray', linestyle=':', alpha=0.7)
        ax.fill_between(future_nums.flatten(), forecast_values*0.85, forecast_values*1.15,
                        alpha=0.15, color='red', label='Confidence Range')
        ax.set_title("Sales Forecast — Next 6 Months", fontsize=14, fontweight='bold')
        ax.set_ylabel("Revenue (₹)")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

        forecast_buf = io.BytesIO()
        fig.savefig(forecast_buf, format='png', dpi=150, bbox_inches='tight')
        forecast_buf.seek(0)
        plt.close()

        forecast_df = pd.DataFrame({
            'Month': [str(m) for m in future_months],
            'Forecasted Revenue': [f"₹{v:,.0f}" for v in forecast_values]
        })
        st.table(forecast_df)

        last_6_avg = monthly['Revenue'].tail(6).mean()
        next_6_avg = forecast_values.mean()
        growth_pct = ((next_6_avg - last_6_avg) / last_6_avg) * 100

        if growth_pct >= 0:
            st.success(f"📈 Revenue trending UPWARD — Expected growth: +{growth_pct:.1f}%")
        else:
            st.warning(f"📉 Revenue trending DOWNWARD — Expected change: {growth_pct:.1f}% — Review strategy!")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 9: SMART ALERTS
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-header'>🚨 Smart Business Alerts</div>", unsafe_allow_html=True)

    alerts = []

    if growth_pct is not None and growth_pct < -10:
        alerts.append(f"⚠️ Revenue declining at {growth_pct:.1f}% — immediate strategy review needed")

    if product_col and qty_col:
        slow_products = df_clean.groupby(product_col)[qty_col].sum().sort_values().head(3)
        for p, q in slow_products.items():
            alerts.append(f"⚠️ Slow moving product: **{p}** — only {q:.0f} units sold")

    if person_col and rev_col:
        person_rev = df_clean.groupby(person_col)[rev_col].sum()
        avg_rev = person_rev.mean()
        below_avg = person_rev[person_rev < avg_rev * 0.7]
        for p in below_avg.index:
            alerts.append(f"⚠️ Salesperson **{p}** is 30%+ below team average")

    if len(alerts) == 0:
        st.markdown("<div class='insight-box'>✅ No critical alerts — business looks healthy!</div>", unsafe_allow_html=True)
    else:
        for alert in alerts:
            st.markdown(f"<div class='alert-box'>{alert}</div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 10: PDF REPORT
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-header'>📄 Download Full Report</div>", unsafe_allow_html=True)

    if st.button("🗒️ Generate PDF Report"):
        pdf_buf = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buf, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
        styles = getSampleStyleSheet()
        story = []

        title_style = ParagraphStyle('T', fontSize=22, textColor=colors.HexColor('#6366f1'),
                                     spaceAfter=6, alignment=1, fontName='Helvetica-Bold')
        sub_style = ParagraphStyle('S', fontSize=12, textColor=colors.grey, spaceAfter=4, alignment=1)
        tag_style = ParagraphStyle('Tag', fontSize=10, textColor=colors.grey, spaceAfter=20, alignment=1)
        section_style = ParagraphStyle('Sec', fontSize=14, textColor=colors.HexColor('#0a0a0a'),
                                       spaceBefore=16, spaceAfter=8, fontName='Helvetica-Bold')

        story.append(Paragraph("Velytics", title_style))
        story.append(Paragraph("Sales Intelligence Report", sub_style))
        story.append(Paragraph("Your data scientist. No hiring required.", tag_style))
        story.append(Paragraph(f"Generated: {pd.Timestamp.now().strftime('%d %B %Y, %I:%M %p')}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))

        # Key Metrics Table
        story.append(Paragraph("Key Metrics", section_style))
        metrics = [["Metric", "Value"]]
        if rev_col:
            metrics.append(["Total Revenue", f"₹{df_clean[rev_col].sum():,.0f}"])
        if target_col:
            metrics.append(["Target Achievement", f"{(df_clean[rev_col].sum()/df_clean[target_col].sum()*100):.1f}%"])
        if qty_col:
            metrics.append(["Total Units Sold", f"{df_clean[qty_col].sum():,.0f}"])
        if region_col:
            metrics.append(["Top Region", df_clean.groupby(region_col)[rev_col].sum().idxmax()])
        if product_col:
            metrics.append(["Top Product", df_clean.groupby(product_col)[rev_col].sum().idxmax()])
        if person_col:
            metrics.append(["Top Salesperson", df_clean.groupby(person_col)[rev_col].sum().idxmax()])
        metrics.append(["Total Records", str(len(df_clean))])
        metrics.append(["Issues Auto-Fixed", str(issues_fixed)])

        t = Table(metrics, colWidths=[2.5*inch, 3.5*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#6366f1')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('PADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(t)

        # Forecast chart
        if forecast_buf:
            story.append(Paragraph("Sales Forecast", section_style))
            forecast_buf.seek(0)
            story.append(RLImage(forecast_buf, width=6*inch, height=3*inch))

        # Alerts
        if alerts:
            story.append(Paragraph("Business Alerts", section_style))
            for alert in alerts:
                clean_alert = alert.replace("**", "").replace("⚠️", "WARNING:")
                story.append(Paragraph(f"• {clean_alert}", styles['Normal']))
                story.append(Spacer(1, 0.05*inch))

        # Recommendation
        story.append(Paragraph("Recommendation", section_style))
        if growth_pct is not None:
            if growth_pct >= 0:
                rec = f"Revenue is trending upward at +{growth_pct:.1f}%. Continue current strategy and focus on top performing regions and products."
            else:
                rec = f"Revenue is trending downward at {growth_pct:.1f}%. Recommend reviewing pricing strategy, focusing on top regions, and supporting underperforming salespersons."
            story.append(Paragraph(rec, styles['Normal']))

        doc.build(story)
        pdf_buf.seek(0)

        st.download_button("📥 Download Velytics Report", data=pdf_buf,
                           file_name="velytics_sales_report.pdf", mime="application/pdf")
        st.success("✅ Report ready!")

else:
    st.markdown("""
    <div style='text-align:center; padding: 48px 20px 32px; background: white; border-radius: 16px; border: 1.5px solid #e5e7eb; margin-top: 16px;'>
        <div style='font-size: 3rem; margin-bottom: 16px;'>⚡</div>
        <h3 style='color:#0a0a0a; font-size:1.5rem; font-weight:800; letter-spacing:-0.5px; margin:0 0 8px;'>Upload your data to get started</h3>
        <p style='color:#6b7280; font-size:1rem; margin:0 0 24px;'>Excel or CSV · Any business data · Results in under 60 seconds</p>
        <div style='display:flex; justify-content:center; gap:32px; flex-wrap:wrap; margin-top:24px;'>
            <div style='text-align:left; max-width:200px;'>
                <div style='font-weight:700; color:#0a0a0a; margin-bottom:8px;'>📊 Instant Analysis</div>
                <div style='color:#6b7280; font-size:0.875rem; line-height:1.7;'>Revenue breakdown · Product intelligence · Price inconsistencies</div>
            </div>
            <div style='text-align:left; max-width:200px;'>
                <div style='font-weight:700; color:#0a0a0a; margin-bottom:8px;'>👤 People Insights</div>
                <div style='color:#6b7280; font-size:0.875rem; line-height:1.7;'>Salesperson rankings · Customer behavior · Payment patterns</div>
            </div>
            <div style='text-align:left; max-width:200px;'>
                <div style='font-weight:700; color:#0a0a0a; margin-bottom:8px;'>🔮 Predictions</div>
                <div style='color:#6b7280; font-size:0.875rem; line-height:1.7;'>6-month forecast · Smart alerts · One-click PDF report</div>
            </div>
        </div>
    </div>
    <p style='text-align:center; color:#9ca3af; font-size:0.82rem; margin-top:16px;'>🔒 Your data is processed and deleted immediately. Never stored. Never shared.</p>
    """, unsafe_allow_html=True)
