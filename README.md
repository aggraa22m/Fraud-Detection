# Financial Fraud Detection System

A comprehensive automated fraud detection system supporting multiple fraud types including credit card fraud, tax evasion, invoice fraud, expense fraud, and GST/HST tax discrepancies. Built with machine learning and advanced pattern recognition.

## Overview

This intelligent system automatically detects the type of financial data uploaded and applies appropriate fraud detection algorithms. It uses **Isolation Forest** machine learning models combined with domain-specific rules to identify anomalies, duplicates, and fraudulent patterns across various financial domains.

## Features

### 🎯 Multi-Type Fraud Detection

**💳 Credit Card Fraud**
- Transaction pattern anomalies using V1-V28 PCA features
- Amount-based anomaly detection
- Temporal pattern analysis

**📊 Tax Evasion Detection**
- Income underreporting identification
- Excessive/unrealistic deduction flagging
- Tax rate anomaly detection
- Round number syndrome detection (fabricated data indicator)
- Business expense fraud detection

**📄 Invoice/Procurement Fraud**
- Duplicate invoice detection
- Split invoice identification (just below approval thresholds)
- Vendor concentration analysis
- Invoice sequence gap detection
- Round amount flagging

**💰 Employee Expense Fraud**
- Duplicate receipt detection
- Just-below-limit expense flagging
- Excessive employee expense identification
- Suspicious timing analysis (weekends/holidays)
- Round number detection

**🇨🇦 GST/HST Tax Fraud**
- Tax calculation validation (5%, 13%, 14%, 15% rates)
- Input tax credit fraud detection
- Duplicate transaction identification
- Tax rate compliance checking

### 🚀 Core Capabilities

- **Auto-Type Detection**: Automatically identifies data type (credit card, tax filing, invoice, expense, GST/HST)
- **ML-Based Anomaly Detection**: Uses Isolation Forest for pattern-based fraud detection
- **Risk Scoring**: Assigns 0-100 risk scores to each record for prioritization
- **Interactive Dashboard**: Modern web-based UI with interactive visualizations
- **Performance Metrics**: Precision, recall, F1-score, ROC AUC (when ground truth available)
- **Export Capabilities**: Download anomalies and full analysis results as CSV

## Tech Stack

- **Python 3.8+**
- **Scikit-learn**: Machine learning (Isolation Forest)
- **Pandas**: Data manipulation and CSV processing
- **Streamlit**: Interactive web dashboard
- **Plotly**: Data visualization and charts
- **NumPy**: Numerical computations

## Installation

1. Clone this repository or download the project files

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Recommended: Unified Multi-Type Fraud Detection (Best Option) ⭐

**For analyzing ANY type of financial data with automatic detection:**

```bash
streamlit run app_unified.py
```

Or double-click `run_unified.bat` (Windows) / `run_unified.sh` (Linux/Mac)

**Supports:**
- Credit card transactions
- Tax filings
- Invoices/procurement
- Employee expenses
- GST/HST records
- General financial data

**Generate Sample Data for Testing:**

```bash
python generate_multi_fraud_samples.py
```

This creates multiple sample files:
- `sample_tax_filing.csv` - Tax evasion examples
- `sample_invoices.csv` - Invoice fraud patterns
- `sample_expenses.csv` - Expense fraud cases
- `sample_gst_hst.csv` - GST/HST fraud examples

### Alternative: Specialized Applications

**Option A: Credit Card Fraud Only**
```bash
streamlit run app_creditcard.py
```
Requires: `creditcard.csv` with Time, V1-V28, Amount, Class columns

**Option B: Canadian GST/HST Tax Analysis Only**
```bash
streamlit run app.py
```
Requires: CSV with subtotal, tax, total, merchant columns

The dashboard will open at `http://localhost:8501` in your default browser.

### 3. Analyze Your Data

1. **Upload CSV File**: Click "Browse files" in the sidebar to upload your financial data
2. **Configure Settings**: Adjust the expected anomaly rate using the slider (default: 10%)
3. **View Results**: Explore the interactive visualizations and analysis across multiple tabs:
   - **Overview**: Risk score distribution and key metrics
   - **Anomaly Analysis**: Scatter plots and top risk transactions
   - **Tax Analysis**: Tax ratio distributions against expected rates
   - **Detailed Results**: Complete table of detected anomalies
4. **Download Results**: Export anomalies or full analysis as CSV files

## Data Format

Your CSV file should contain the following columns:

| Column | Description | Required |
|--------|-------------|----------|
| `transaction_id` | Unique transaction identifier | Yes |
| `date` | Transaction date/time | Yes |
| `merchant` | Merchant or vendor name | Recommended |
| `subtotal` | Pre-tax amount | Yes |
| `tax` | GST/HST amount | Yes |
| `total` | Total transaction amount | Yes |
| `province` | Canadian province code | Optional |
| `category` | Transaction category | Optional |

### Example Data Format

```csv
transaction_id,date,merchant,province,subtotal,tax,total,category
TXN000001,2024-01-15 10:30:00,ABC Electronics,ON,100.00,13.00,113.00,Electronics
TXN000002,2024-01-16 14:22:00,XYZ Grocery,BC,50.00,2.50,52.50,Food
```

## How It Works

### 1. Data Processing
- Loads transaction data from CSV files
- Validates and cleans data
- Engineers features for analysis

### 2. Duplicate Detection
- Identifies transactions with matching subtotal, tax, merchant, and amount
- Flags potential duplicate entries for review

### 3. Tax Validation
- Calculates tax-to-subtotal ratios
- Compares against expected Canadian tax rates:
  - **GST**: 5% (AB, BC, MB, QC, SK, NT, NU, YT)
  - **HST**: 13% (ON), 15% (NS, NB, NL, PE)
- Flags transactions with anomalous tax calculations

### 4. Machine Learning Analysis
- **Isolation Forest** algorithm identifies outliers in transaction patterns
- Features considered:
  - Transaction amounts
  - Tax ratios
  - Merchant frequency
  - Temporal patterns (day of week, time of day)
  - Amount z-scores
- Assigns anomaly scores and risk ratings

### 5. Risk Scoring
- Converts anomaly scores to 0-100 risk scale
- Categories:
  - **High Risk**: Score > 75
  - **Medium Risk**: Score 50-75
  - **Low Risk**: Score < 50

## Project Structure

```
Fraud detection/
├── app_unified.py                    # Unified multi-type fraud dashboard ⭐
├── unified_fraud_detector.py         # Core multi-type fraud detection engine
├── generate_multi_fraud_samples.py   # Multi-type sample data generator
│
├── app_creditcard.py                 # Credit card specific dashboard
├── creditcard_adapter.py             # Credit card fraud detector
│
├── app.py                            # GST/HST tax fraud dashboard
├── fraud_detector.py                 # GST/HST fraud detector
├── generate_sample_data.py           # GST/HST sample generator
│
├── requirements.txt                  # Python dependencies
├── README.md                         # Documentation
│
├── run_unified.bat/sh               # Unified system launcher ⭐
├── run_creditcard.bat/sh            # Credit card launcher
└── run.bat/sh                       # GST/HST launcher
```

## Configuration

The fraud detection model can be configured with:

- **Contamination Rate**: Expected proportion of anomalies (adjustable in UI, default: 0.1)
- **Random State**: Set to 42 for reproducible results
- **Number of Estimators**: 100 trees in Isolation Forest ensemble
- **Feature Engineering**: Customizable feature selection

## Performance Considerations

- Handles datasets with thousands of transactions efficiently
- Parallel processing enabled for model training (`n_jobs=-1`)
- Automatic scaling of features for optimal detection
- Memory-efficient CSV processing with Pandas

## Use Cases

- **Financial Auditing**: Detect irregularities in expense reports or invoices
- **Compliance Monitoring**: Ensure correct tax calculations for Canadian provinces
- **Fraud Prevention**: Identify suspicious transaction patterns
- **Data Quality**: Find duplicate entries and data entry errors
- **Risk Assessment**: Prioritize transactions for manual review

## Limitations

- Designed specifically for Canadian GST/HST tax structures
- Requires historical transaction data for pattern learning
- Performance depends on data quality and feature availability
- Anomaly detection accuracy improves with larger datasets

## Future Enhancements

Potential improvements for future versions:

- Real-time transaction monitoring
- Advanced feature engineering (merchant reputation, geographic patterns)
- Integration with accounting software APIs
- Custom rule engine for domain-specific fraud patterns
- Multi-model ensemble approaches
- Email notifications for high-risk detections

## License

This project is available for personal and educational use.

## Support

For issues or questions about using this fraud detection system:

1. Check that your CSV file matches the expected format
2. Ensure all required columns are present
3. Verify Python version compatibility (3.8+)
4. Confirm all dependencies are installed correctly

## Acknowledgments

Built using open-source machine learning and data visualization libraries. Designed for Canadian financial compliance and fraud detection use cases.
