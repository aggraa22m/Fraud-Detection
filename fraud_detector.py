"""
Financial Anomaly Detection System
Detects duplicate entries and tax discrepancies in Canadian financial records
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class FraudDetector:
    """
    Automated fraud detection system for financial transactions.
    Uses Isolation Forest to identify anomalies in transaction patterns.
    """

    def __init__(self, contamination=0.1, random_state=42):
        """
        Initialize the fraud detector.

        Args:
            contamination: Expected proportion of outliers in the dataset
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load financial data from CSV file."""
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} transactions from {filepath}")
        return df

    def detect_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify potential duplicate transactions.

        Args:
            df: DataFrame containing transaction data

        Returns:
            DataFrame with duplicate flag
        """
        # Define columns to check for duplicates (excluding transaction ID)
        dup_columns = [col for col in df.columns if col not in ['transaction_id', 'date']]

        df['is_duplicate'] = df.duplicated(subset=dup_columns, keep=False)
        return df

    def calculate_tax_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate GST/HST ratios and identify tax discrepancies.
        Standard Canadian tax rates: GST=5%, HST=13-15% depending on province
        """
        if 'subtotal' in df.columns and 'tax' in df.columns:
            df['tax_ratio'] = (df['tax'] / df['subtotal'] * 100).round(2)

            # Flag transactions with unusual tax ratios
            # Expected ranges: 5% (GST) or 13-15% (HST)
            expected_rates = [5.0, 13.0, 14.0, 15.0]
            tolerance = 0.5

            def check_tax_anomaly(ratio):
                if pd.isna(ratio):
                    return True
                return not any(abs(ratio - rate) <= tolerance for rate in expected_rates)

            df['tax_anomaly'] = df['tax_ratio'].apply(check_tax_anomaly)

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for anomaly detection model.
        """
        # Ensure required columns exist
        feature_df = df.copy()

        # Transaction frequency by merchant/vendor
        if 'merchant' in df.columns:
            feature_df['merchant_frequency'] = df.groupby('merchant')['merchant'].transform('count')

        # Amount statistics
        if 'amount' in df.columns or 'total' in df.columns:
            amount_col = 'amount' if 'amount' in df.columns else 'total'
            feature_df['amount_zscore'] = (df[amount_col] - df[amount_col].mean()) / df[amount_col].std()

        # Time-based features
        if 'date' in df.columns:
            feature_df['date'] = pd.to_datetime(df['date'])
            feature_df['day_of_week'] = feature_df['date'].dt.dayofweek
            feature_df['hour'] = feature_df['date'].dt.hour if hasattr(feature_df['date'].dt, 'hour') else 12

        return feature_df

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare numerical features for the Isolation Forest model.
        """
        numeric_features = []

        # Select numerical columns
        potential_features = ['subtotal', 'tax', 'total', 'amount', 'tax_ratio',
                            'merchant_frequency', 'amount_zscore', 'day_of_week', 'hour']

        for col in potential_features:
            if col in df.columns:
                numeric_features.append(col)

        self.feature_columns = numeric_features

        if not numeric_features:
            raise ValueError("No numerical features found for modeling")

        X = df[numeric_features].fillna(0)
        return X

    def fit_predict(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Fit Isolation Forest model and predict anomalies.

        Returns:
            Tuple of (DataFrame with predictions, statistics dictionary)
        """
        # Engineer features
        df = self.engineer_features(df)

        # Prepare feature matrix
        X = self.prepare_features(df)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Initialize and fit Isolation Forest
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1
        )

        # Predict anomalies (-1 for outliers, 1 for inliers)
        predictions = self.model.fit_predict(X_scaled)
        anomaly_scores = self.model.score_samples(X_scaled)

        # Add predictions to dataframe
        df['anomaly'] = predictions
        df['anomaly_score'] = anomaly_scores
        df['is_anomaly'] = df['anomaly'] == -1

        # Calculate risk score (0-100 scale)
        df['risk_score'] = ((1 - (anomaly_scores - anomaly_scores.min()) /
                            (anomaly_scores.max() - anomaly_scores.min())) * 100).round(2)

        # Generate statistics
        stats = {
            'total_transactions': len(df),
            'anomalies_detected': int(df['is_anomaly'].sum()),
            'anomaly_rate': float(df['is_anomaly'].mean() * 100),
            'duplicates_found': int(df['is_duplicate'].sum()) if 'is_duplicate' in df.columns else 0,
            'tax_anomalies': int(df['tax_anomaly'].sum()) if 'tax_anomaly' in df.columns else 0,
            'high_risk_transactions': int((df['risk_score'] > 75).sum()),
            'features_used': self.feature_columns
        }

        return df, stats

    def get_top_anomalies(self, df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        """Get top N most anomalous transactions."""
        return df.nlargest(n, 'risk_score')

    def analyze(self, filepath: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete analysis pipeline: load data, detect duplicates,
        calculate tax ratios, and identify anomalies.

        Args:
            filepath: Path to CSV file containing transaction data

        Returns:
            Tuple of (analyzed DataFrame, statistics dictionary)
        """
        # Load data
        df = self.load_data(filepath)

        # Detect duplicates
        df = self.detect_duplicates(df)

        # Calculate tax ratios and anomalies
        df = self.calculate_tax_ratio(df)

        # Fit model and predict anomalies
        df, stats = self.fit_predict(df)

        return df, stats


if __name__ == "__main__":
    # Example usage
    detector = FraudDetector(contamination=0.1)

    # For testing with sample data
    print("Fraud Detection System initialized")
    print("Use the Streamlit UI to analyze your financial data")
