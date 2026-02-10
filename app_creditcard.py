"""
Credit Card Fraud Detection Dashboard
Interactive Streamlit UI for credit card transaction analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from creditcard_adapter import CreditCardFraudDetector
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


def display_header():
    """Display application header."""
    st.markdown('<p class="main-header">💳 Credit Card Fraud Detection System</p>', unsafe_allow_html=True)
    st.markdown("---")


def display_metrics(stats):
    """Display key metrics."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📊 Total Transactions",
            value=f"{stats['total_transactions']:,}"
        )

    with col2:
        st.metric(
            label="🚨 Detected Anomalies",
            value=f"{stats['anomalies_detected']:,}",
            delta=f"{stats['anomaly_rate']:.2f}%"
        )

    with col3:
        st.metric(
            label="⚠️ High Risk (>75)",
            value=f"{stats['high_risk_transactions']:,}"
        )

    with col4:
        if 'actual_frauds' in stats:
            st.metric(
                label="✓ Actual Frauds",
                value=f"{stats['actual_frauds']:,}",
                delta=f"{stats['actual_fraud_rate']:.2f}%"
            )
        else:
            st.metric(
                label="🔧 Features Used",
                value=stats['features_used']
            )


def display_performance_metrics(stats):
    """Display ML performance metrics if available."""
    if 'precision' not in stats:
        return

    st.markdown("### 📈 Model Performance Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Precision", f"{stats['precision']:.3f}")
    with col2:
        st.metric("Recall", f"{stats['recall']:.3f}")
    with col3:
        st.metric("F1 Score", f"{stats['f1_score']:.3f}")
    with col4:
        st.metric("Accuracy", f"{stats['accuracy']:.3f}")
    with col5:
        st.metric("ROC AUC", f"{stats['roc_auc']:.3f}")

    # Confusion Matrix
    st.markdown("#### Confusion Matrix")
    col1, col2 = st.columns(2)

    with col1:
        # Create confusion matrix visualization
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
            textfont={"size": 20},
            colorscale='Blues',
            showscale=False
        ))

        fig.update_layout(
            title='Confusion Matrix',
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Performance breakdown
        st.markdown("**Detection Breakdown:**")
        st.write(f"✅ **True Positives:** {stats['true_positives']:,} (Correctly identified frauds)")
        st.write(f"❌ **False Positives:** {stats['false_positives']:,} (Normal flagged as fraud)")
        st.write(f"✅ **True Negatives:** {stats['true_negatives']:,} (Correctly identified normal)")
        st.write(f"❌ **False Negatives:** {stats['false_negatives']:,} (Missed frauds)")

        st.markdown("---")
        detection_rate = (stats['true_positives'] / stats['actual_frauds'] * 100) if stats['actual_frauds'] > 0 else 0
        st.write(f"**Detection Rate:** {detection_rate:.1f}% of actual frauds detected")


def plot_risk_distribution(df):
    """Plot risk score distribution."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Risk Score Distribution', 'Risk by Fraud Status'),
        specs=[[{'type': 'histogram'}, {'type': 'box'}]]
    )

    # Histogram
    fig.add_trace(
        go.Histogram(x=df['risk_score'], nbinsx=50, name='Risk Score', marker_color='#1f77b4'),
        row=1, col=1
    )

    # Box plot by actual fraud status if available
    if 'actual_fraud' in df.columns:
        for fraud_status in [0, 1]:
            data = df[df['actual_fraud'] == fraud_status]['risk_score']
            name = 'Fraud' if fraud_status == 1 else 'Normal'
            color = '#ff4444' if fraud_status == 1 else '#44ff44'

            fig.add_trace(
                go.Box(y=data, name=name, marker_color=color),
                row=1, col=2
            )

    fig.update_xaxes(title_text="Risk Score", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Risk Score", row=1, col=2)

    fig.add_hline(y=75, line_dash="dash", line_color="red", row=1, col=2,
                  annotation_text="High Risk")

    fig.update_layout(height=400, showlegend=True, margin=dict(l=20, r=20, t=40, b=20))

    return fig


def plot_amount_distribution(df):
    """Plot transaction amount distributions."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Amount Distribution (Log Scale)', 'Amount vs Risk Score')
    )

    # Log-scale histogram
    amounts = df['Amount'][df['Amount'] > 0]  # Filter out zero amounts
    fig.add_trace(
        go.Histogram(x=np.log10(amounts + 1), nbinsx=50, name='Amount', marker_color='#2ca02c'),
        row=1, col=1
    )

    # Scatter plot
    if 'actual_fraud' in df.columns:
        for fraud_status in [0, 1]:
            data = df[df['actual_fraud'] == fraud_status]
            name = 'Fraud' if fraud_status == 1 else 'Normal'
            color = '#ff4444' if fraud_status == 1 else '#44ff44'

            fig.add_trace(
                go.Scatter(
                    x=data['Amount'],
                    y=data['risk_score'],
                    mode='markers',
                    name=name,
                    marker=dict(color=color, size=5, opacity=0.5)
                ),
                row=1, col=2
            )

    fig.update_xaxes(title_text="Log10(Amount)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_xaxes(title_text="Transaction Amount ($)", row=1, col=2)
    fig.update_yaxes(title_text="Risk Score", row=1, col=2)

    fig.update_layout(height=400, showlegend=True, margin=dict(l=20, r=20, t=40, b=20))

    return fig


def plot_temporal_analysis(df):
    """Plot temporal patterns."""
    if 'hour' not in df.columns:
        return None

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Transactions by Hour', 'Fraud Rate by Hour')
    )

    # Transactions by hour
    hourly_counts = df.groupby('hour').size()
    fig.add_trace(
        go.Bar(x=hourly_counts.index, y=hourly_counts.values, name='Transactions', marker_color='#1f77b4'),
        row=1, col=1
    )

    # Fraud rate by hour if labels available
    if 'actual_fraud' in df.columns:
        fraud_rate = df.groupby('hour')['actual_fraud'].mean() * 100
        fig.add_trace(
            go.Scatter(x=fraud_rate.index, y=fraud_rate.values, mode='lines+markers',
                      name='Fraud Rate %', marker_color='#ff4444', line=dict(width=3)),
            row=1, col=2
        )

    fig.update_xaxes(title_text="Hour of Day", row=1, col=1)
    fig.update_yaxes(title_text="Transaction Count", row=1, col=1)
    fig.update_xaxes(title_text="Hour of Day", row=1, col=2)
    fig.update_yaxes(title_text="Fraud Rate (%)", row=1, col=2)

    fig.update_layout(height=400, showlegend=True, margin=dict(l=20, r=20, t=40, b=20))

    return fig


def plot_feature_importance(df):
    """Plot top V features correlation with fraud."""
    if 'actual_fraud' not in df.columns:
        return None

    # Calculate correlation of V features with fraud
    v_cols = [col for col in df.columns if col.startswith('V')]
    correlations = df[v_cols + ['actual_fraud']].corr()['actual_fraud'].drop('actual_fraud')
    correlations = correlations.abs().sort_values(ascending=False).head(15)

    fig = go.Figure(data=[
        go.Bar(x=correlations.values, y=correlations.index, orientation='h', marker_color='#9467bd')
    ])

    fig.update_layout(
        title='Top 15 Features Correlated with Fraud',
        xaxis_title='Absolute Correlation',
        yaxis_title='Feature',
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


def display_anomaly_table(df, max_rows=100):
    """Display detected anomalies table."""
    anomalies = df[df['is_anomaly'] == True].sort_values('risk_score', ascending=False).head(max_rows)

    if len(anomalies) == 0:
        st.info("No anomalies detected with current settings.")
        return

    st.subheader(f"🚨 Top {len(anomalies)} Detected Anomalies")

    # Select columns to display
    display_cols = ['transaction_id', 'Amount', 'risk_score']
    if 'datetime' in anomalies.columns:
        display_cols.insert(1, 'datetime')
    if 'actual_fraud' in anomalies.columns:
        display_cols.append('actual_fraud')

    # Add some V features
    v_cols = [f'V{i}' for i in range(1, 6)]
    for col in v_cols:
        if col in anomalies.columns:
            display_cols.append(col)

    st.dataframe(
        anomalies[display_cols],
        use_container_width=True,
        height=400
    )

    # Download button
    csv = anomalies.to_csv(index=False)
    st.download_button(
        label="📥 Download Detected Anomalies",
        data=csv,
        file_name="credit_card_anomalies.csv",
        mime="text/csv"
    )


def main():
    """Main application."""
    display_header()

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    # Sample size for large datasets
    use_sample = st.sidebar.checkbox("Use Sample (faster)", value=True,
                                     help="Use a sample of data for faster processing")

    if use_sample:
        sample_size = st.sidebar.slider("Sample Size", 1000, 50000, 10000, 1000)
    else:
        sample_size = None

    # Contamination rate
    contamination = st.sidebar.slider(
        "Expected Fraud Rate (%)",
        min_value=0.1,
        max_value=10.0,
        value=0.5,
        step=0.1,
        help="Expected percentage of fraudulent transactions"
    ) / 100

    # Check if creditcard.csv exists
    import os
    file_path = "creditcard.csv"

    if os.path.exists(file_path):
        if st.sidebar.button("🚀 Run Analysis", type="primary"):
            try:
                with st.spinner("🔄 Analyzing credit card transactions..."):
                    detector = CreditCardFraudDetector(contamination=contamination)
                    df_results, stats = detector.analyze(file_path, sample_size=sample_size)

                st.success("✅ Analysis Complete!")

                # Display metrics
                display_metrics(stats)

                # Performance metrics
                if 'precision' in stats:
                    st.markdown("---")
                    display_performance_metrics(stats)

                st.markdown("---")

                # Tabs for different visualizations
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Risk Analysis",
                    "💰 Amount Analysis",
                    "⏰ Temporal Patterns",
                    "🔍 Feature Analysis",
                    "📋 Detailed Results"
                ])

                with tab1:
                    st.plotly_chart(plot_risk_distribution(df_results), use_container_width=True)

                with tab2:
                    st.plotly_chart(plot_amount_distribution(df_results), use_container_width=True)

                with tab3:
                    temporal_fig = plot_temporal_analysis(df_results)
                    if temporal_fig:
                        st.plotly_chart(temporal_fig, use_container_width=True)
                    else:
                        st.info("Temporal analysis not available")

                with tab4:
                    feature_fig = plot_feature_importance(df_results)
                    if feature_fig:
                        st.plotly_chart(feature_fig, use_container_width=True)
                        st.info("Features V1-V28 are PCA-transformed components. Higher correlation indicates stronger relationship with fraud.")
                    else:
                        st.info("Feature analysis requires actual fraud labels")

                with tab5:
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
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    else:
        st.warning("⚠️ creditcard.csv not found!")
        st.info("""
        **Expected file:** `creditcard.csv` in the project directory

        This dataset should contain:
        - Time, V1-V28 (PCA features), Amount, Class columns
        - Downloadable from Kaggle or other sources
        """)

        st.markdown("### 📚 About This System")
        st.markdown("""
        This fraud detection system uses **Isolation Forest** machine learning to identify
        anomalous credit card transactions based on:

        - **Amount patterns**: Unusual transaction amounts
        - **PCA features**: V1-V28 anonymized characteristics
        - **Temporal patterns**: Time-based behavior
        - **Risk scoring**: 0-100 scale for prioritization

        Upload your `creditcard.csv` file and click **Run Analysis** to begin!
        """)


if __name__ == "__main__":
    main()
