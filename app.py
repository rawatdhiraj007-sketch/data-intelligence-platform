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

        # Auto Clean
        df_inv = df.copy()
        issues_fixed_inv = 0
        for col in df_inv.select_dtypes(include='number').columns:
            missing = df_inv[col].isnull().sum()
            if missing > 0:
                df_inv[col].fillna(df_inv[col].median(), inplace=True)
                issues_fixed_inv += missing
        df_inv.drop_duplicates(inplace=True)

        # Detect columns
        prod_col   = next((c for c in df_inv.columns if 'product name' in c.lower() or 'product' in c.lower()), None)
        cat_col    = next((c for c in df_inv.columns if 'category' in c.lower()), None)
        stock_col  = next((c for c in df_inv.columns if 'current stock' in c.lower() or 'stock' in c.lower()), None)
        reorder_col= next((c for c in df_inv.columns if 'reorder' in c.lower()), None)
        max_col    = next((c for c in df_inv.columns if 'max stock' in c.lower()), None)
        cost_col   = next((c for c in df_inv.columns if 'unit cost' in c.lower() or 'cost' in c.lower()), None)
        price_col  = next((c for c in df_inv.columns if 'selling price' in c.lower() or 'price' in c.lower()), None)
        demand_col = next((c for c in df_inv.columns if 'monthly demand' in c.lower() or 'demand' in c.lower()), None)
        days_col   = next((c for c in df_inv.columns if 'days in stock' in c.lower()), None)
        sold30_col = next((c for c in df_inv.columns if 'last 30' in c.lower()), None)
        sold90_col = next((c for c in df_inv.columns if 'last 90' in c.lower()), None)
        last_sold_col = next((c for c in df_inv.columns if 'last sold' in c.lower()), None)
        ware_col   = next((c for c in df_inv.columns if 'warehouse' in c.lower()), None)
        supplier_col = next((c for c in df_inv.columns if 'supplier' in c.lower()), None)

        st.success(f"✅ Inventory file loaded — {len(df_inv):,} products · {len(df_inv.columns)} columns · {issues_fixed_inv} issues auto-fixed")

        # ── SECTION 1: KEY INVENTORY METRICS ────────────────────────────────
        st.markdown("<div class='section-header'>📦 Inventory Overview</div>", unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        if stock_col:
            total_stock = df_inv[stock_col].sum()
            c1.metric("Total Stock Units", f"{total_stock:,.0f}")
        if prod_col:
            c2.metric("Total SKUs", f"{df_inv[prod_col].nunique()}")
        if reorder_col and stock_col:
            below_reorder = (df_inv[stock_col] <= df_inv[reorder_col]).sum()
            c3.metric("Items Below Reorder", f"{below_reorder}", delta=f"Need restocking", delta_color="inverse")
        if stock_col:
            out_of_stock = (df_inv[stock_col] == 0).sum()
            c4.metric("Out of Stock", f"{out_of_stock}", delta="Critical", delta_color="inverse")
        if cost_col and stock_col:
            inventory_value = (df_inv[stock_col] * df_inv[cost_col]).sum()
            c5.metric("Total Inventory Value", f"${inventory_value:,.0f}")

        # ── SECTION 2: STOCK STATUS ANALYSIS ────────────────────────────────
        if stock_col and reorder_col:
            st.markdown("<div class='section-header'>🚦 Stock Status Analysis</div>", unsafe_allow_html=True)

            df_inv['Stock Status'] = 'Normal'
            df_inv.loc[df_inv[stock_col] == 0, 'Stock Status'] = 'Out of Stock'
            df_inv.loc[(df_inv[stock_col] > 0) & (df_inv[stock_col] <= df_inv[reorder_col]), 'Stock Status'] = 'Low Stock'
            if max_col in df_inv.columns:
                df_inv.loc[df_inv[stock_col] >= df_inv[max_col] * 0.9, 'Stock Status'] = 'Overstocked'

            col_a, col_b = st.columns(2)
            with col_a:
                status_counts = df_inv['Stock Status'].value_counts()
                fig, ax = plt.subplots(figsize=(6, 4))
                colors_map = {'Out of Stock': '#ef4444', 'Low Stock': '#f59e0b', 'Normal': '#10b981', 'Overstocked': '#6366f1'}
                bar_colors = [colors_map.get(s, '#6366f1') for s in status_counts.index]
                bars = ax.bar(status_counts.index, status_counts.values, color=bar_colors)
                ax.set_ylabel("Number of Products")
                ax.set_title("Stock Status Distribution", fontweight='bold')
                for bar, val in zip(bars, status_counts.values):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), str(val), ha='center', va='bottom', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig); plt.close()

            with col_b:
                st.write("**Products Needing Immediate Action**")
                urgent = df_inv[df_inv['Stock Status'].isin(['Out of Stock', 'Low Stock'])]
                if len(urgent) > 0:
                    show_cols = [c for c in [prod_col, cat_col, stock_col, reorder_col, 'Stock Status'] if c]
                    st.dataframe(urgent[show_cols].sort_values('Stock Status').reset_index(drop=True), use_container_width=True)
                else:
                    st.markdown("<div class='insight-box'>✅ All products have healthy stock levels!</div>", unsafe_allow_html=True)

        # ── SECTION 3: DEAD STOCK DETECTION ─────────────────────────────────
        st.markdown("<div class='section-header'>💀 Dead Stock Detection</div>", unsafe_allow_html=True)

        dead_stock = pd.DataFrame()
        if last_sold_col and stock_col and prod_col:
            dead_stock = df_inv[(df_inv[last_sold_col] > 90) & (df_inv[stock_col] > 0)].copy()
            if cost_col in df_inv.columns:
                dead_stock['Capital Locked ($)'] = (dead_stock[stock_col] * dead_stock[cost_col]).round(0)

        if len(dead_stock) > 0:
            total_locked = dead_stock['Capital Locked ($)'].sum() if 'Capital Locked ($)' in dead_stock.columns else 0
            st.markdown(f"<div class='alert-box'>⚠️ <b>{len(dead_stock)} products</b> haven't sold in 90+ days — <b>${total_locked:,.0f}</b> capital locked in dead stock!</div>", unsafe_allow_html=True)
            show_cols = [c for c in [prod_col, cat_col, stock_col, last_sold_col, 'Capital Locked ($)'] if c in dead_stock.columns]
            st.dataframe(dead_stock[show_cols].sort_values('Capital Locked ($)', ascending=False).reset_index(drop=True), use_container_width=True)
        else:
            st.markdown("<div class='insight-box'>✅ No dead stock detected — inventory is moving well!</div>", unsafe_allow_html=True)

        # ── SECTION 4: ABC ANALYSIS ──────────────────────────────────────────
        if prod_col and sold90_col:
            st.markdown("<div class='section-header'>🏆 ABC Analysis — Product Prioritization</div>", unsafe_allow_html=True)
            st.caption("A = Top 20% products driving 80% of sales · B = Middle 30% · C = Bottom 50% (slow movers)")

            abc = df_inv.groupby(prod_col)[sold90_col].sum().sort_values(ascending=False).reset_index()
            abc.columns = ['Product', 'Units Sold (90 Days)']
            abc['Cumulative %'] = (abc['Units Sold (90 Days)'].cumsum() / abc['Units Sold (90 Days)'].sum() * 100).round(1)
            abc['ABC Class'] = 'C'
            abc.loc[abc['Cumulative %'] <= 80, 'ABC Class'] = 'A'
            abc.loc[(abc['Cumulative %'] > 80) & (abc['Cumulative %'] <= 95), 'ABC Class'] = 'B'

            col_a, col_b = st.columns(2)
            with col_a:
                abc_counts = abc['ABC Class'].value_counts()
                fig, ax = plt.subplots(figsize=(5, 4))
                abc_colors = {'A': '#10b981', 'B': '#6366f1', 'C': '#9ca3af'}
                wedge_colors = [abc_colors.get(c, '#6366f1') for c in abc_counts.index]
                ax.pie(abc_counts.values, labels=[f"Class {c}" for c in abc_counts.index],
                       autopct='%1.0f%%', colors=wedge_colors, startangle=90)
                ax.set_title("ABC Distribution", fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig); plt.close()

            with col_b:
                st.write("**Class A — Star Products (Protect These)**")
                class_a = abc[abc['ABC Class'] == 'A'].head(10)
                st.dataframe(class_a[['Product', 'Units Sold (90 Days)', 'ABC Class']].reset_index(drop=True), use_container_width=True)

        # ── SECTION 5: REORDER ALERTS ────────────────────────────────────────
        if stock_col and reorder_col and demand_col and prod_col:
            st.markdown("<div class='section-header'>🔔 Reorder Intelligence</div>", unsafe_allow_html=True)

            df_inv['Days of Stock Left'] = (df_inv[stock_col] / (df_inv[demand_col] / 30)).round(1)
            df_inv['Days of Stock Left'] = df_inv['Days of Stock Left'].replace([float('inf'), -float('inf')], 999)

            reorder_needed = df_inv[df_inv['Days of Stock Left'] < 30].sort_values('Days of Stock Left')

            if len(reorder_needed) > 0:
                st.markdown(f"<div class='alert-box'>⚠️ <b>{len(reorder_needed)} products</b> will run out within 30 days — reorder now!</div>", unsafe_allow_html=True)
                show_cols = [c for c in [prod_col, cat_col, stock_col, demand_col, 'Days of Stock Left', supplier_col] if c and c in reorder_needed.columns]
                display_df = reorder_needed[show_cols].reset_index(drop=True)
                st.dataframe(display_df, use_container_width=True)
            else:
                st.markdown("<div class='insight-box'>✅ All products have sufficient stock for next 30 days.</div>", unsafe_allow_html=True)

        # ── SECTION 6: CATEGORY PERFORMANCE ─────────────────────────────────
        if cat_col and stock_col:
            st.markdown("<div class='section-header'>📊 Category Intelligence</div>", unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Stock Value by Category**")
                if cost_col:
                    df_inv['Stock Value'] = df_inv[stock_col] * df_inv[cost_col]
                    cat_value = df_inv.groupby(cat_col)['Stock Value'].sum().sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    cat_value.plot(kind='barh', ax=ax, color='#6366f1')
                    ax.set_xlabel("Stock Value ($)")
                    ax.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig); plt.close()

            with col_b:
                st.write("**Units Sold (Last 90 Days) by Category**")
                if sold90_col:
                    cat_sold = df_inv.groupby(cat_col)[sold90_col].sum().sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.pie(cat_sold.values, labels=cat_sold.index, autopct='%1.1f%%',
                           colors=['#6366f1','#818cf8','#a5b4fc','#c7d2fe','#e0e7ff','#4f46e5'])
                    plt.tight_layout()
                    st.pyplot(fig); plt.close()

        # ── SECTION 7: PROFIT MARGIN ANALYSIS ───────────────────────────────
        if cost_col and price_col and prod_col:
            st.markdown("<div class='section-header'>💰 Profit Margin Analysis</div>", unsafe_allow_html=True)

            df_inv['Margin ($)'] = df_inv[price_col] - df_inv[cost_col]
            df_inv['Margin %'] = ((df_inv['Margin ($)'] / df_inv[price_col]) * 100).round(1)

            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Top 10 Highest Margin Products**")
                top_margin = df_inv[[prod_col, cost_col, price_col, 'Margin %']].sort_values('Margin %', ascending=False).head(10)
                top_margin = top_margin.rename(columns={cost_col: 'Cost ($)', price_col: 'Price ($)'})
                st.dataframe(top_margin.reset_index(drop=True), use_container_width=True)

            with col_b:
                st.write("**Bottom 10 Lowest Margin Products**")
                low_margin = df_inv[[prod_col, cost_col, price_col, 'Margin %']].sort_values('Margin %').head(10)
                low_margin = low_margin.rename(columns={cost_col: 'Cost ($)', price_col: 'Price ($)'})
                st.dataframe(low_margin.reset_index(drop=True), use_container_width=True)

        # ── SECTION 8: SMART INVENTORY ALERTS ───────────────────────────────
        st.markdown("<div class='section-header'>🚨 Smart Inventory Alerts</div>", unsafe_allow_html=True)

        inv_alerts = []
        if stock_col:
            oos = (df_inv[stock_col] == 0).sum()
            if oos > 0:
                inv_alerts.append(f"🔴 <b>{oos} products</b> are completely OUT OF STOCK — immediate action required")
        if 'Days of Stock Left' in df_inv.columns:
            critical = (df_inv['Days of Stock Left'] < 7).sum()
            if critical > 0:
                inv_alerts.append(f"🔴 <b>{critical} products</b> will run out within 7 days — URGENT reorder needed")
        if len(dead_stock) > 0:
            inv_alerts.append(f"⚠️ <b>{len(dead_stock)} products</b> are dead stock — consider discounting or liquidating")
        if 'Stock Status' in df_inv.columns:
            over = (df_inv['Stock Status'] == 'Overstocked').sum()
            if over > 0:
                inv_alerts.append(f"⚠️ <b>{over} products</b> are overstocked — capital tied up unnecessarily")

        if len(inv_alerts) == 0:
            st.markdown("<div class='insight-box'>✅ Inventory looks healthy — no critical alerts!</div>", unsafe_allow_html=True)
        else:
            for alert in inv_alerts:
                st.markdown(f"<div class='alert-box'>{alert}</div>", unsafe_allow_html=True)

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
