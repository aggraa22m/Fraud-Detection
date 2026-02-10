"""
Unified Financial Fraud Detection Dashboard
Supports multiple fraud types with auto-detection
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from unified_fraud_detector import UnifiedFraudDetector
import numpy as np

# Page config
st.set_page_config(
    page_title="Financial Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .fraud-type-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem;
    }
    .credit-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
    .tax-filing { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }
    .invoice { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; }
    .expense { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white; }
    .gst-hst { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white; }
    </style>
""", unsafe_allow_html=True)


def display_header(data_type=None):
    """Display application header with data type."""
    st.markdown('<p class="main-header">🔍 Financial Fraud Detection System</p>', unsafe_allow_html=True)

    if data_type:
        type_labels = {
            'credit_card': '💳 Credit Card Fraud',
            'tax_filing': '📊 Tax Evasion Detection',
            'invoice': '📄 Invoice/Procurement Fraud',
            'expense': '💰 Expense Fraud',
            'gst_hst': '🇨🇦 GST/HST Tax Fraud',
            'general': '📋 General Financial Fraud'
        }

        type_classes = {
            'credit_card': 'credit-card',
            'tax_filing': 'tax-filing',
            'invoice': 'invoice',
            'expense': 'expense',
            'gst_hst': 'gst-hst',
            'general': 'invoice'
        }

        label = type_labels.get(data_type, 'Unknown')
        css_class = type_classes.get(data_type, 'invoice')

        st.markdown(f'<div class="fraud-type-badge {css_class}">Detected Type: {label}</div>',
                   unsafe_allow_html=True)

    st.markdown("---")


def display_metrics(stats):
    """Display key metrics based on data type."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Total Records", f"{stats['total_records']:,}")

    with col2:
        st.metric("🚨 Anomalies Detected", f"{stats['anomalies_detected']:,}",
                 delta=f"{stats['anomaly_rate']:.2f}%")

    with col3:
        st.metric("⚠️ High Risk", f"{stats['high_risk_count']:,}")

    with col4:
        st.metric("✅ Features Used", stats['features_used'])

    # Data-type specific metrics
    st.markdown("---")
    st.subheader("📌 Fraud-Specific Findings")

    cols = st.columns(4)

    if stats['data_type'] == 'tax_filing':
        with cols[0]:
            if 'tax_rate_anomalies' in stats:
                st.metric("Tax Rate Anomalies", f"{stats.get('tax_rate_anomalies', 0):,}")
        with cols[1]:
            if 'excessive_deductions' in stats:
                st.metric("Excessive Deductions", f"{stats.get('excessive_deductions', 0):,}")
        with cols[2]:
            if 'income_underreporting' in stats:
                st.metric("Income Underreporting", f"{stats.get('income_underreporting', 0):,}")

    elif stats['data_type'] == 'invoice':
        with cols[0]:
            if 'duplicate_invoices' in stats:
                st.metric("Duplicate Invoices", f"{stats.get('duplicate_invoices', 0):,}")
        with cols[1]:
            if 'split_invoices' in stats:
                st.metric("Split Invoices", f"{stats.get('split_invoices', 0):,}")
        with cols[2]:
            if 'vendor_concentration_issues' in stats:
                st.metric("Vendor Issues", f"{stats.get('vendor_concentration_issues', 0):,}")

    elif stats['data_type'] == 'gst_hst':
        with cols[0]:
            if 'tax_calculation_errors' in stats:
                st.metric("Tax Calc Errors", f"{stats.get('tax_calculation_errors', 0):,}")
        with cols[1]:
            if 'duplicate_transactions' in stats:
                st.metric("Duplicates", f"{stats.get('duplicate_transactions', 0):,}")

    elif stats['data_type'] == 'expense':
        with cols[0]:
            if 'duplicate_receipts' in stats:
                st.metric("Duplicate Receipts", f"{stats.get('duplicate_receipts', 0):,}")
        with cols[1]:
            if 'suspicious_amounts' in stats:
                st.metric("Suspicious Amounts", f"{stats.get('suspicious_amounts', 0):,}")

    # Performance metrics if available
    if 'precision' in stats:
        st.markdown("---")
        st.subheader("🎯 Model Performance")
        perf_cols = st.columns(5)
        with perf_cols[0]:
            st.metric("Precision", f"{stats['precision']:.3f}")
        with perf_cols[1]:
            st.metric("Recall", f"{stats['recall']:.3f}")
        with perf_cols[2]:
            st.metric("F1 Score", f"{stats['f1_score']:.3f}")
        with perf_cols[3]:
            st.metric("Accuracy", f"{stats['accuracy']:.3f}")
        with perf_cols[4]:
            st.metric("Actual Frauds", f"{stats['actual_frauds']:,}")


def plot_risk_distribution(df):
    """Plot risk score distribution."""
    fig = px.histogram(
        df,
        x='risk_score',
        nbins=50,
        title='Risk Score Distribution',
        labels={'risk_score': 'Risk Score', 'count': 'Frequency'},
        color_discrete_sequence=['#667eea']
    )

    fig.add_vline(x=75, line_dash="dash", line_color="red",
                  annotation_text="High Risk Threshold", annotation_position="top right")
    fig.add_vline(x=50, line_dash="dash", line_color="orange",
                  annotation_text="Medium Risk", annotation_position="top left")

    fig.update_layout(showlegend=False, height=400, margin=dict(l=20, r=20, t=40, b=20))

    return fig


def plot_top_features(df, stats, n=10):
    """Plot top features by importance (correlation with anomaly)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    exclude = ['anomaly', 'anomaly_score', 'risk_score', 'is_anomaly']
    feature_cols = [col for col in numeric_cols if col not in exclude]

    if not feature_cols:
        return None

    # Calculate correlation with risk score
    correlations = {}
    for col in feature_cols[:20]:  # Limit to first 20
        try:
            corr = abs(df[col].corr(df['risk_score']))
            if not np.isnan(corr):
                correlations[col] = corr
        except:
            pass

    if not correlations:
        return None

    # Sort and get top N
    top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:n]

    if not top_features:
        return None

    features, values = zip(*top_features)

    fig = go.Figure(data=[
        go.Bar(x=list(values), y=list(features), orientation='h',
               marker_color='#764ba2')
    ])

    fig.update_layout(
        title=f'Top {n} Features Correlated with Fraud Risk',
        xaxis_title='Absolute Correlation with Risk Score',
        yaxis_title='Feature',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


def plot_anomaly_breakdown(stats):
    """Plot anomaly breakdown by risk level."""
    labels = ['High Risk (>75)', 'Medium Risk (50-75)', 'Low Risk (<50)']
    values = [
        stats['high_risk_count'],
        stats['medium_risk_count'],
        stats['low_risk_count']
    ]
    colors = ['#f5576c', '#ffa726', '#66bb6a']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors
    )])

    fig.update_layout(
        title='Risk Level Distribution',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


def plot_confusion_matrix(stats):
    """Plot confusion matrix if ground truth available."""
    if 'precision' not in stats:
        return None

    cm_data = [
        [stats['true_negatives'], stats['false_positives']],
        [stats['false_negatives'], stats['true_positives']]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=cm_data,
        x=['Predicted Normal', 'Predicted Fraud'],
        y=['Actual Normal', 'Actual Fraud'],
        text=cm_data,
        texttemplate='%{text}',
        textfont={"size": 24},
        colorscale='RdYlGn_r',
        showscale=False
    ))

    fig.update_layout(
        title='Confusion Matrix',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


def display_anomaly_table(df, max_rows=100):
    """Display table of detected anomalies."""
    anomalies = df[df['is_anomaly'] == True].sort_values('risk_score', ascending=False)

    if len(anomalies) == 0:
        st.info("✅ No anomalies detected with current settings.")
        return

    st.subheader(f"🚨 Top {min(len(anomalies), max_rows)} Detected Anomalies")

    # Select relevant columns
    display_cols = ['risk_score']

    # Add common columns if they exist
    common_cols = ['amount', 'total', 'vendor', 'merchant', 'employee',
                   'income', 'deductions', 'tax', 'date', 'invoice_number']

    for col in anomalies.columns:
        if col.lower() in common_cols or col in common_cols:
            display_cols.append(col)

    # Limit columns
    display_cols = display_cols[:10]

    display_df = anomalies.head(max_rows)[display_cols]

    st.dataframe(display_df, use_container_width=True, height=400)

    # Download button
    csv = anomalies.to_csv(index=False)
    st.download_button(
        label="📥 Download All Anomalies",
        data=csv,
        file_name="fraud_anomalies.csv",
        mime="text/csv"
    )


def main():
    """Main application."""

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Financial Data (CSV)",
        type=['csv'],
        help="Upload any type of financial data - system will auto-detect the type"
    )

    sample_size = None
    if st.sidebar.checkbox("Use Sample (faster for large files)", value=False):
        sample_size = st.sidebar.slider("Sample Size", 1000, 50000, 10000, 1000)

    contamination = st.sidebar.slider(
        "Expected Fraud Rate (%)",
        min_value=0.1,
        max_value=20.0,
        value=5.0,
        step=0.1,
        help="Expected percentage of fraudulent records"
    ) / 100

    if uploaded_file is None:
        # Welcome screen
        display_header()

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            ## Welcome to the Unified Financial Fraud Detection System

            This advanced system automatically detects and analyzes multiple types of financial fraud:

            ### 🎯 Supported Fraud Types

            **💳 Credit Card Fraud**
            - Transaction pattern anomalies
            - Unusual spending behavior
            - ML-based fraud detection

            **📊 Tax Evasion**
            - Income underreporting
            - Excessive/unrealistic deductions
            - Tax rate anomalies
            - Round number syndrome

            **📄 Invoice/Procurement Fraud**
            - Duplicate invoices
            - Split invoices (just below approval limits)
            - Vendor concentration issues
            - Invoice sequence gaps

            **💰 Expense Fraud**
            - Duplicate receipts
            - Just-below-limit expenses
            - Excessive employee expenses
            - Suspicious timing (weekends/holidays)

            **🇨🇦 GST/HST Tax Fraud**
            - Tax calculation errors
            - Input tax credit fraud
            - Duplicate transactions
            - Rate validation (5%, 13%, 14%, 15%)

            ### 🚀 How It Works

            1. **Upload** your CSV file (any financial data type)
            2. **Auto-detection** identifies the data type automatically
            3. **Analysis** applies appropriate fraud detection algorithms
            4. **Results** interactive visualizations and downloadable reports

            ### 📤 Get Started

            👈 Upload your CSV file in the sidebar to begin analysis!
            """)

        with col2:
            st.info("""
            **📋 Sample Data Formats**

            **Credit Card:**
            `Time, V1-V28, Amount, Class`

            **Tax Filing:**
            `Income, Deductions, Tax_Paid, Taxable_Income`

            **Invoice:**
            `Invoice_Number, Vendor, Amount, Date`

            **Expense:**
            `Employee, Amount, Category, Date, Merchant`

            **GST/HST:**
            `Subtotal, Tax, Total, Merchant, Province`
            """)

            st.success("""
            **✨ Features:**
            - Auto-type detection
            - ML anomaly detection
            - Risk scoring (0-100)
            - Interactive dashboards
            - Export capabilities
            """)

    else:
        # Analysis mode
        if st.sidebar.button("🚀 Run Fraud Analysis", type="primary"):
            try:
                # Save uploaded file
                with open("temp_upload.csv", "wb") as f:
                    f.write(uploaded_file.getvalue())

                with st.spinner("🔄 Analyzing financial data..."):
                    # Initialize detector
                    detector = UnifiedFraudDetector(contamination=contamination)

                    # Run analysis
                    df_results, stats = detector.analyze("temp_upload.csv", sample_size=sample_size)

                st.success("✅ Fraud Analysis Complete!")

                # Display header with detected type
                display_header(stats['data_type'])

                # Metrics
                display_metrics(stats)

                st.markdown("---")

                # Tabs
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Overview",
                    "🔍 Detailed Analysis",
                    "📈 Feature Insights",
                    "📋 Anomaly Records"
                ])

                with tab1:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.plotly_chart(plot_risk_distribution(df_results), use_container_width=True)

                    with col2:
                        st.plotly_chart(plot_anomaly_breakdown(stats), use_container_width=True)

                    # Confusion matrix if available
                    cm_fig = plot_confusion_matrix(stats)
                    if cm_fig:
                        st.plotly_chart(cm_fig, use_container_width=True)

                with tab2:
                    st.subheader("Risk Analysis by Data Type")

                    # Show top anomalous records summary
                    top_risks = df_results.nlargest(10, 'risk_score')

                    st.write("**Top 10 Highest Risk Records:**")
                    st.dataframe(top_risks[['risk_score'] + [c for c in top_risks.columns if c in ['amount', 'total', 'income', 'vendor', 'employee']][:5]],
                               use_container_width=True)

                with tab3:
                    feature_fig = plot_top_features(df_results, stats)
                    if feature_fig:
                        st.plotly_chart(feature_fig, use_container_width=True)
                    else:
                        st.info("Feature analysis not available for this dataset")

                    with st.expander("📋 All Features Used"):
                        st.write(stats['feature_names'])

                with tab4:
                    display_anomaly_table(df_results)

                    st.markdown("---")
                    st.subheader("📥 Download Complete Results")
                    csv_full = df_results.to_csv(index=False)
                    st.download_button(
                        label="Download Full Analysis",
                        data=csv_full,
                        file_name="fraud_analysis_complete.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
