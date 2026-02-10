"""
Fraud Detection Dashboard
Interactive Streamlit UI for visualizing financial anomalies
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fraud_detector import FraudDetector
import io

# Page configuration
st.set_page_config(
    page_title="Financial Anomaly Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .anomaly-high {
        background-color: #ffebee;
        padding: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .anomaly-medium {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-left: 4px solid #ff9800;
    }
    .anomaly-low {
        background-color: #e8f5e9;
        padding: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)


def display_header():
    """Display application header."""
    st.markdown('<p class="main-header">🔍 Financial Anomaly Detector</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    **Automated fraud detection system** for identifying duplicate entries and tax discrepancies
    in Canadian financial records using machine learning.
    """)


def display_metrics(stats):
    """Display key metrics in columns."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Transactions",
            value=f"{stats['total_transactions']:,}"
        )

    with col2:
        st.metric(
            label="Anomalies Detected",
            value=f"{stats['anomalies_detected']:,}",
            delta=f"{stats['anomaly_rate']:.1f}%"
        )

    with col3:
        st.metric(
            label="Duplicate Entries",
            value=f"{stats['duplicates_found']:,}"
        )

    with col4:
        st.metric(
            label="Tax Discrepancies",
            value=f"{stats['tax_anomalies']:,}"
        )


def plot_risk_distribution(df):
    """Create risk score distribution plot."""
    fig = px.histogram(
        df,
        x='risk_score',
        nbins=50,
        title='Risk Score Distribution',
        labels={'risk_score': 'Risk Score', 'count': 'Number of Transactions'},
        color_discrete_sequence=['#1f77b4']
    )

    fig.add_vline(x=75, line_dash="dash", line_color="red",
                  annotation_text="High Risk Threshold")

    fig.update_layout(
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


def plot_anomaly_scatter(df):
    """Create scatter plot of anomalies."""
    # Select appropriate columns for visualization
    amount_col = 'total' if 'total' in df.columns else ('amount' if 'amount' in df.columns else 'subtotal')
    tax_col = 'tax' if 'tax' in df.columns else None

    if amount_col and tax_col:
        fig = px.scatter(
            df,
            x=amount_col,
            y=tax_col,
            color='risk_score',
            size='risk_score',
            hover_data=['is_anomaly', 'is_duplicate'] if 'is_duplicate' in df.columns else ['is_anomaly'],
            title='Transaction Amount vs Tax (Colored by Risk)',
            labels={amount_col: 'Transaction Amount ($)', tax_col: 'Tax Amount ($)'},
            color_continuous_scale='RdYlGn_r'
        )
    else:
        # Fallback visualization
        fig = px.scatter(
            df.reset_index(),
            x='index',
            y='risk_score',
            color='is_anomaly',
            title='Transaction Risk Scores',
            labels={'index': 'Transaction Index', 'risk_score': 'Risk Score'},
            color_discrete_map={True: 'red', False: 'green'}
        )

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


def plot_tax_ratio_distribution(df):
    """Create tax ratio distribution plot."""
    if 'tax_ratio' not in df.columns:
        return None

    fig = go.Figure()

    # Plot distribution
    fig.add_trace(go.Histogram(
        x=df['tax_ratio'],
        nbinsx=50,
        name='Tax Ratios',
        marker_color='#1f77b4'
    ))

    # Add expected tax rate lines
    expected_rates = [5.0, 13.0, 14.0, 15.0]
    colors = ['green', 'blue', 'purple', 'orange']

    for rate, color in zip(expected_rates, colors):
        fig.add_vline(
            x=rate,
            line_dash="dash",
            line_color=color,
            annotation_text=f"{rate}%"
        )

    fig.update_layout(
        title='Tax Ratio Distribution (GST/HST Rates)',
        xaxis_title='Tax Ratio (%)',
        yaxis_title='Number of Transactions',
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


def plot_top_anomalies(df, n=10):
    """Create bar chart of top anomalies."""
    top_anomalies = df.nlargest(n, 'risk_score')

    # Create identifier for x-axis
    if 'merchant' in top_anomalies.columns:
        x_label = top_anomalies['merchant'].astype(str)
    elif 'transaction_id' in top_anomalies.columns:
        x_label = top_anomalies['transaction_id'].astype(str)
    else:
        x_label = top_anomalies.index.astype(str)

    fig = px.bar(
        top_anomalies,
        x=x_label,
        y='risk_score',
        title=f'Top {n} Highest Risk Transactions',
        labels={'x': 'Transaction', 'risk_score': 'Risk Score'},
        color='risk_score',
        color_continuous_scale='Reds'
    )

    fig.update_layout(
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_tickangle=-45
    )

    return fig


def display_anomaly_table(df, max_rows=100):
    """Display table of anomalous transactions."""
    anomalies = df[df['is_anomaly'] == True].sort_values('risk_score', ascending=False)

    if len(anomalies) == 0:
        st.info("No anomalies detected in the dataset.")
        return

    st.subheader(f"🚨 Detected Anomalies ({len(anomalies)} total)")

    # Display top anomalies
    display_df = anomalies.head(max_rows).copy()

    # Format risk score with color
    def risk_color(score):
        if score >= 75:
            return 'background-color: #ffcdd2'
        elif score >= 50:
            return 'background-color: #fff9c4'
        else:
            return 'background-color: #c8e6c9'

    # Select columns to display
    display_columns = ['risk_score', 'is_duplicate', 'tax_anomaly'] if 'tax_anomaly' in display_df.columns else ['risk_score', 'is_duplicate']

    # Add amount columns if available
    for col in ['total', 'amount', 'subtotal', 'tax', 'merchant', 'date']:
        if col in display_df.columns:
            display_columns.append(col)

    st.dataframe(
        display_df[display_columns].style.applymap(
            risk_color, subset=['risk_score']
        ),
        use_container_width=True,
        height=400
    )

    # Download button for anomalies
    csv = anomalies.to_csv(index=False)
    st.download_button(
        label="📥 Download Anomalies as CSV",
        data=csv,
        file_name="detected_anomalies.csv",
        mime="text/csv"
    )


def main():
    """Main application function."""
    display_header()

    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")

    contamination = st.sidebar.slider(
        "Expected Anomaly Rate (%)",
        min_value=1,
        max_value=30,
        value=10,
        help="Expected percentage of anomalies in your data"
    ) / 100

    # File upload
    st.sidebar.header("📁 Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload financial transaction data in CSV format"
    )

    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            with open("temp_data.csv", "wb") as f:
                f.write(uploaded_file.getvalue())

            # Initialize detector
            with st.spinner("🔄 Analyzing transactions..."):
                detector = FraudDetector(contamination=contamination)
                df_results, stats = detector.analyze("temp_data.csv")

            # Display results
            st.success("✅ Analysis Complete!")

            # Show metrics
            display_metrics(stats)

            st.markdown("---")

            # Visualizations
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Overview",
                "🎯 Anomaly Analysis",
                "💰 Tax Analysis",
                "📋 Detailed Results"
            ])

            with tab1:
                st.subheader("Risk Score Distribution")
                st.plotly_chart(plot_risk_distribution(df_results), use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("High Risk Transactions", f"{stats['high_risk_transactions']:,}")
                with col2:
                    st.metric("Features Used", len(stats['features_used']))

                with st.expander("View Features Used"):
                    st.write(stats['features_used'])

            with tab2:
                st.subheader("Anomaly Visualization")
                st.plotly_chart(plot_anomaly_scatter(df_results), use_container_width=True)

                st.subheader("Top Risk Transactions")
                st.plotly_chart(plot_top_anomalies(df_results), use_container_width=True)

            with tab3:
                st.subheader("Tax Ratio Analysis")
                tax_fig = plot_tax_ratio_distribution(df_results)
                if tax_fig:
                    st.plotly_chart(tax_fig, use_container_width=True)

                    st.info("""
                    **Expected Canadian Tax Rates:**
                    - 🟢 GST: 5%
                    - 🔵 HST: 13% (ON), 14% (PEI), 15% (NS, NB, NL)
                    """)
                else:
                    st.warning("Tax ratio data not available in the uploaded file.")

            with tab4:
                display_anomaly_table(df_results)

                st.markdown("---")
                st.subheader("📥 Download Full Results")

                csv_full = df_results.to_csv(index=False)
                st.download_button(
                    label="Download Complete Analysis",
                    data=csv_full,
                    file_name="fraud_detection_results.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV contains the required columns (e.g., subtotal, tax, total, merchant, date)")

    else:
        # Welcome screen
        st.info("👈 Upload a CSV file to begin analysis")

        st.markdown("### 📋 Expected CSV Format")
        st.markdown("""
        Your CSV file should contain financial transaction data with columns such as:
        - `transaction_id` - Unique identifier
        - `date` - Transaction date
        - `merchant` - Merchant/vendor name
        - `subtotal` - Pre-tax amount
        - `tax` - GST/HST amount
        - `total` - Total amount
        """)

        # Sample data preview
        st.markdown("### 📝 Sample Data Format")
        sample_data = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN002', 'TXN003'],
            'date': ['2024-01-15', '2024-01-16', '2024-01-17'],
            'merchant': ['Store A', 'Store B', 'Store C'],
            'subtotal': [100.00, 200.00, 150.00],
            'tax': [5.00, 26.00, 7.50],
            'total': [105.00, 226.00, 157.50]
        })
        st.dataframe(sample_data, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🎯 What This Tool Does")
        st.markdown("""
        - ✅ Detects duplicate transactions
        - ✅ Identifies tax discrepancies (GST/HST anomalies)
        - ✅ Uses machine learning (Isolation Forest) to find unusual patterns
        - ✅ Assigns risk scores to each transaction
        - ✅ Provides interactive visualizations
        """)


if __name__ == "__main__":
    main()
