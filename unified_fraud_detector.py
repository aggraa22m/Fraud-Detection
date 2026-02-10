"""
Unified Financial Fraud Detection System
Supports multiple fraud types: credit card, tax evasion, invoice fraud, expense fraud
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class UnifiedFraudDetector:
    """
    Comprehensive fraud detection system that auto-detects data type
    and applies appropriate fraud detection algorithms.
    """

    def __init__(self, contamination=0.1, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.data_type = None
        self.feature_columns = []

    def detect_data_type(self, df: pd.DataFrame) -> str:
        """
        Auto-detect the type of financial data.

        Returns:
            'credit_card', 'tax_filing', 'invoice', 'expense', 'general'
        """
        columns = set(df.columns.str.lower())

        # Credit card detection
        v_features = sum(1 for col in df.columns if col.startswith('V') and col[1:].isdigit())
        if v_features >= 10 and 'amount' in columns:
            return 'credit_card'

        # Tax filing detection
        tax_indicators = {'income', 'deductions', 'tax_paid', 'taxable_income', 'credits'}
        if len(tax_indicators.intersection(columns)) >= 2:
            return 'tax_filing'

        # Invoice/Expense detection
        invoice_indicators = {'invoice', 'vendor', 'supplier', 'purchase_order', 'po_number'}
        expense_indicators = {'expense', 'category', 'department', 'employee'}

        if len(invoice_indicators.intersection(columns)) >= 2:
            return 'invoice'
        elif len(expense_indicators.intersection(columns)) >= 2:
            return 'expense'

        # GST/HST tax detection
        gst_indicators = {'gst', 'hst', 'subtotal', 'tax', 'total'}
        if len(gst_indicators.intersection(columns)) >= 3:
            return 'gst_hst'

        return 'general'

    def load_data(self, filepath: str, sample_size: int = None) -> pd.DataFrame:
        """Load financial data and detect type."""
        df = pd.read_csv(filepath)

        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=self.random_state)

        self.data_type = self.detect_data_type(df)
        print(f"Detected data type: {self.data_type}")
        print(f"Loaded {len(df)} records")

        return df

    def detect_tax_evasion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect tax evasion patterns in tax filing data.
        """
        df = df.copy()

        # Normalize column names
        df.columns = df.columns.str.lower()

        # 1. Income underreporting detection
        if 'income' in df.columns and 'reported_income' in df.columns:
            df['income_variance'] = abs(df['income'] - df['reported_income']) / df['income']
            df['income_underreported'] = df['income_variance'] > 0.1

        # 2. Excessive deductions
        if 'deductions' in df.columns and 'income' in df.columns:
            df['deduction_ratio'] = df['deductions'] / df['income']
            # Flag deductions > 50% of income as suspicious
            df['excessive_deductions'] = df['deduction_ratio'] > 0.5

        # 3. Tax rate anomalies
        if 'tax_paid' in df.columns and 'taxable_income' in df.columns:
            df['effective_tax_rate'] = (df['tax_paid'] / df['taxable_income'] * 100).replace([np.inf, -np.inf], 0)

            # Expected progressive tax rates
            expected_rates = df['taxable_income'].apply(self._calculate_expected_tax_rate)
            df['tax_rate_deviation'] = abs(df['effective_tax_rate'] - expected_rates)
            df['tax_rate_anomaly'] = df['tax_rate_deviation'] > 5  # More than 5% deviation

        # 4. Unrealistic expense claims
        if 'business_expenses' in df.columns and 'income' in df.columns:
            df['expense_ratio'] = df['business_expenses'] / df['income']
            df['unrealistic_expenses'] = df['expense_ratio'] > 0.7

        # 5. Round number syndrome (fabricated numbers)
        if 'income' in df.columns:
            df['income_rounded'] = df['income'].apply(lambda x: x % 1000 == 0 or x % 500 == 0)

        if 'deductions' in df.columns:
            df['deductions_rounded'] = df['deductions'].apply(lambda x: x % 100 == 0)

        return df

    def _calculate_expected_tax_rate(self, income: float) -> float:
        """Calculate expected tax rate based on income brackets."""
        if income <= 50000:
            return 15.0
        elif income <= 100000:
            return 20.5
        elif income <= 150000:
            return 26.0
        elif income <= 200000:
            return 29.0
        else:
            return 33.0

    def detect_invoice_fraud(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect invoice and procurement fraud.
        """
        df = df.copy()
        df.columns = df.columns.str.lower()

        # 1. Duplicate invoices
        dup_columns = ['amount', 'vendor', 'description'] if all(c in df.columns for c in ['amount', 'vendor', 'description']) else None
        if dup_columns:
            df['is_duplicate'] = df.duplicated(subset=dup_columns, keep=False)

        # 2. Split invoices (just below approval threshold)
        if 'amount' in df.columns:
            # Common thresholds: 1000, 5000, 10000
            thresholds = [1000, 5000, 10000]
            df['near_threshold'] = False

            for threshold in thresholds:
                # Invoices within 5% below threshold
                near = (df['amount'] >= threshold * 0.95) & (df['amount'] < threshold)
                df['near_threshold'] = df['near_threshold'] | near

        # 3. Vendor concentration (same vendor too frequently)
        if 'vendor' in df.columns:
            vendor_counts = df['vendor'].value_counts()
            df['vendor_frequency'] = df['vendor'].map(vendor_counts)
            total = len(df)
            df['vendor_concentration'] = (df['vendor_frequency'] / total * 100).round(2)
            df['high_vendor_concentration'] = df['vendor_concentration'] > 20  # >20% from same vendor

        # 4. Round number invoices (suspicious)
        if 'amount' in df.columns:
            df['round_amount'] = df['amount'].apply(lambda x: x % 100 == 0 or x % 1000 == 0)

        # 5. Invoice number sequence gaps
        if 'invoice_number' in df.columns:
            df['invoice_number_numeric'] = pd.to_numeric(df['invoice_number'], errors='coerce')
            df_sorted = df.sort_values('invoice_number_numeric')
            df_sorted['invoice_gap'] = df_sorted['invoice_number_numeric'].diff()
            # Large gaps might indicate missing/deleted invoices
            df['large_invoice_gap'] = df_sorted['invoice_gap'] > 10

        return df

    def detect_gst_hst_fraud(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect GST/HST tax fraud in Canadian transactions.
        """
        df = df.copy()
        df.columns = df.columns.str.lower()

        # Calculate tax ratio
        if 'subtotal' in df.columns and 'tax' in df.columns:
            df['tax_ratio'] = (df['tax'] / df['subtotal'] * 100).replace([np.inf, -np.inf], 0).round(2)

            # Expected Canadian tax rates
            expected_rates = [5.0, 13.0, 14.0, 15.0]  # GST, HST variants
            tolerance = 0.5

            def check_tax_anomaly(ratio):
                if pd.isna(ratio) or ratio == 0:
                    return True
                return not any(abs(ratio - rate) <= tolerance for rate in expected_rates)

            df['tax_anomaly'] = df['tax_ratio'].apply(check_tax_anomaly)

        # Check for input tax credit fraud
        if 'tax_collected' in df.columns and 'tax_paid' in df.columns:
            df['net_tax'] = df['tax_collected'] - df['tax_paid']
            # Suspicious if always claiming refunds
            df['excessive_credits'] = df['net_tax'] < 0

        # Duplicate detection
        dup_cols = [c for c in ['amount', 'vendor', 'date'] if c in df.columns]
        if len(dup_cols) >= 2:
            df['is_duplicate'] = df.duplicated(subset=dup_cols, keep=False)

        return df

    def detect_expense_fraud(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect employee expense fraud.
        """
        df = df.copy()
        df.columns = df.columns.str.lower()

        # 1. Excessive expenses by employee
        if 'employee' in df.columns and 'amount' in df.columns:
            emp_totals = df.groupby('employee')['amount'].sum()
            avg_total = emp_totals.mean()
            df['employee_total'] = df['employee'].map(emp_totals)
            df['excessive_employee_expense'] = df['employee_total'] > avg_total * 2

        # 2. Duplicate receipts
        dup_cols = [c for c in ['amount', 'date', 'merchant'] if c in df.columns]
        if len(dup_cols) >= 2:
            df['duplicate_receipt'] = df.duplicated(subset=dup_cols, keep=False)

        # 3. Just-below-limit expenses
        if 'amount' in df.columns:
            limits = [50, 100, 200, 500]  # Common expense limits
            df['just_below_limit'] = False
            for limit in limits:
                near = (df['amount'] >= limit * 0.95) & (df['amount'] < limit)
                df['just_below_limit'] = df['just_below_limit'] | near

        # 4. Weekend/holiday expenses (suspicious timing)
        if 'date' in df.columns:
            df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
            df['is_weekend'] = df['date_parsed'].dt.dayofweek >= 5

        # 5. Round numbers
        if 'amount' in df.columns:
            df['round_amount'] = df['amount'].apply(lambda x: x % 10 == 0 or x % 50 == 0)

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply appropriate fraud detection based on data type."""

        if self.data_type == 'tax_filing':
            df = self.detect_tax_evasion(df)
        elif self.data_type == 'invoice':
            df = self.detect_invoice_fraud(df)
        elif self.data_type == 'gst_hst':
            df = self.detect_gst_hst_fraud(df)
        elif self.data_type == 'expense':
            df = self.detect_expense_fraud(df)
        elif self.data_type == 'credit_card':
            # Already handled by V features
            pass

        return df

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features based on data type."""

        # Auto-select numerical features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude certain columns
        exclude = ['class', 'fraud', 'is_fraud', 'actual_fraud', 'anomaly', 'risk_score',
                   'anomaly_score', 'is_anomaly']

        feature_cols = [col for col in numeric_cols if col.lower() not in exclude]

        if not feature_cols:
            raise ValueError("No numerical features found for modeling")

        self.feature_columns = feature_cols
        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

        return X

    def fit_predict(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Fit model and predict anomalies with comprehensive statistics.
        """
        # Apply fraud-specific detection
        df = self.engineer_features(df)

        # Prepare features
        X = self.prepare_features(df)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit Isolation Forest
        print(f"Training Isolation Forest with {len(self.feature_columns)} features...")
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1,
            verbose=0
        )

        predictions = self.model.fit_predict(X_scaled)
        anomaly_scores = self.model.score_samples(X_scaled)

        # Add predictions
        df['anomaly'] = predictions
        df['anomaly_score'] = anomaly_scores
        df['is_anomaly'] = df['anomaly'] == -1

        # Risk score (0-100)
        min_score = anomaly_scores.min()
        max_score = anomaly_scores.max()
        df['risk_score'] = ((1 - (anomaly_scores - min_score) / (max_score - min_score)) * 100).round(2)

        # Calculate statistics
        stats = self._calculate_statistics(df)

        return df, stats

    def _calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive statistics."""
        stats = {
            'data_type': self.data_type,
            'total_records': len(df),
            'anomalies_detected': int(df['is_anomaly'].sum()),
            'anomaly_rate': float(df['is_anomaly'].mean() * 100),
            'high_risk_count': int((df['risk_score'] > 75).sum()),
            'medium_risk_count': int(((df['risk_score'] >= 50) & (df['risk_score'] <= 75)).sum()),
            'low_risk_count': int((df['risk_score'] < 50).sum()),
            'features_used': len(self.feature_columns),
            'feature_names': self.feature_columns
        }

        # Data-type specific stats
        if self.data_type == 'tax_filing':
            if 'tax_rate_anomaly' in df.columns:
                stats['tax_rate_anomalies'] = int(df['tax_rate_anomaly'].sum())
            if 'excessive_deductions' in df.columns:
                stats['excessive_deductions'] = int(df['excessive_deductions'].sum())
            if 'income_underreported' in df.columns:
                stats['income_underreporting'] = int(df['income_underreported'].sum())

        elif self.data_type == 'invoice':
            if 'is_duplicate' in df.columns:
                stats['duplicate_invoices'] = int(df['is_duplicate'].sum())
            if 'near_threshold' in df.columns:
                stats['split_invoices'] = int(df['near_threshold'].sum())
            if 'high_vendor_concentration' in df.columns:
                stats['vendor_concentration_issues'] = int(df['high_vendor_concentration'].sum())

        elif self.data_type == 'gst_hst':
            if 'tax_anomaly' in df.columns:
                stats['tax_calculation_errors'] = int(df['tax_anomaly'].sum())
            if 'is_duplicate' in df.columns:
                stats['duplicate_transactions'] = int(df['is_duplicate'].sum())

        elif self.data_type == 'expense':
            if 'duplicate_receipt' in df.columns:
                stats['duplicate_receipts'] = int(df['duplicate_receipt'].sum())
            if 'just_below_limit' in df.columns:
                stats['suspicious_amounts'] = int(df['just_below_limit'].sum())

        # Ground truth comparison if available
        if 'class' in df.columns or 'fraud' in df.columns or 'is_fraud' in df.columns:
            label_col = 'class' if 'class' in df.columns else ('fraud' if 'fraud' in df.columns else 'is_fraud')
            y_true = df[label_col]
            y_pred = df['is_anomaly'].astype(int)

            from sklearn.metrics import confusion_matrix, classification_report

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            stats.update({
                'actual_frauds': int(y_true.sum()),
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn),
                'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
            })

            if stats['precision'] > 0 and stats['recall'] > 0:
                stats['f1_score'] = 2 * stats['precision'] * stats['recall'] / (stats['precision'] + stats['recall'])
            else:
                stats['f1_score'] = 0.0

        return stats

    def analyze(self, filepath: str, sample_size: int = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete fraud analysis pipeline.
        """
        df = self.load_data(filepath, sample_size)
        df, stats = self.fit_predict(df)
        return df, stats


if __name__ == "__main__":
    print("Unified Financial Fraud Detection System")
    print("Supports: Credit Card, Tax Filing, Invoice, Expense, GST/HST")
