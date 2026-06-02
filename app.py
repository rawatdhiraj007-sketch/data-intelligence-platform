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
st.set_page_config(page_title="Data Intelligence Platform", page_icon="📊", layout="wide")

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main { background-color: #f8f9fa; }
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border-left: 4px solid #1F4E79;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin: 8px 0;
}
.section-header {
    background: linear-gradient(90deg, #1F4E79, #2E75B6);
    color: white;
    padding: 12px 20px;
    border-radius: 8px;
    margin: 20px 0 15px 0;
    font-size: 18px;
    font-weight: bold;
}
.alert-box {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 12px 16px;
    border-radius: 6px;
    margin: 6px 0;
}
.insight-box {
    background: #d4edda;
    border-left: 4px solid #28a745;
    padding: 12px 16px;
    border-radius: 6px;
    margin: 6px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 30px 0 10px 0;'>
    <h1 style='color:#1F4E79; font-size:2.5rem; margin:0;'>📊 Data Intelligence Platform</h1>
    <p style='color:#555; font-size:1.1rem; margin-top:8px;'>Upload any business data → Get instant insights like a data scientist</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Upload ───────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📁 Upload your Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file is not None:

    # Load
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

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

    st.success(f"✅ File loaded & cleaned — {len(df_clean)} rows, {len(df_clean.columns)} columns, {issues_fixed} issues auto-fixed")

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
                              color=['#1F4E79','#2E75B6','#9DC3E6','#BDD7EE'])
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
                       colors=['#1F4E79','#2E75B6','#9DC3E6','#BDD7EE','#DEEAF1'])
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
            top_products.plot(kind='barh', ax=ax, color='#2E75B6')
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
                top_qty.plot(kind='barh', ax=ax, color='#1F4E79')
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
            bars = ax.bar(person_data[person_col], person_data['Revenue'], color='#1F4E79')
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
                   colors=['#1F4E79','#2E75B6','#9DC3E6','#BDD7EE'])
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

        title_style = ParagraphStyle('T', fontSize=22, textColor=colors.HexColor('#1F4E79'),
                                     spaceAfter=6, alignment=1, fontName='Helvetica-Bold')
        sub_style = ParagraphStyle('S', fontSize=12, textColor=colors.grey, spaceAfter=20, alignment=1)
        section_style = ParagraphStyle('Sec', fontSize=14, textColor=colors.HexColor('#1F4E79'),
                                       spaceBefore=16, spaceAfter=8, fontName='Helvetica-Bold')

        story.append(Paragraph("📊 Data Intelligence Platform", title_style))
        story.append(Paragraph("Sales Intelligence Report", sub_style))
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
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1F4E79')),
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

        st.download_button("📥 Download PDF Report", data=pdf_buf,
                           file_name="sales_intelligence_report.pdf", mime="application/pdf")
        st.success("✅ Report ready!")

else:
    st.markdown("""
    <div style='text-align:center; padding: 40px;'>
        <h3 style='color:#1F4E79;'>👆 Upload your Excel or CSV file to get started</h3>
        <p style='color:#666; font-size:1rem;'>Supports any sales data format</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **📊 What you get:**
        - Revenue analysis
        - Product/SKU intelligence
        - Price inconsistency alerts
        """)
    with col2:
        st.markdown("""
        **👤 People insights:**
        - Salesperson performance
        - Customer intelligence
        - Payment behavior
        """)
    with col3:
        st.markdown("""
        **🔮 Predictions:**
        - 6-month sales forecast
        - Smart business alerts
        - PDF report download
        """)

    st.markdown("<br><p style='text-align:center; color:#888;'>No data science knowledge needed. Your data is never stored.</p>", unsafe_allow_html=True)
