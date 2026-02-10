"""
Credit Card Dataset Adapter
Converts the Kaggle credit card fraud dataset to work with the fraud detection system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class CreditCardFraudDetector:
    """
    Fraud detection system specifically designed for credit card transaction data.
    Works with anonymized features (V1-V28) from PCA transformation.
    """

    def __init__(self, contamination=0.1, random_state=42):
        """
        Initialize the credit card fraud detector.

        Args:
            contamination: Expected proportion of outliers in the dataset
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()

    def load_data(self, filepath: str, sample_size: int = None) -> pd.DataFrame:
        """
        Load credit card transaction data from CSV file.

        Args:
            filepath: Path to CSV file
            sample_size: Optional - number of rows to sample (for large datasets)
        """
        df = pd.read_csv(filepath)

        if sample_size and sample_size < len(df):
            # Stratified sampling to maintain fraud ratio
            fraud_df = df[df['Class'] == 1]
            normal_df = df[df['Class'] == 0].sample(n=sample_size - len(fraud_df), random_state=self.random_state)
            df = pd.concat([normal_df, fraud_df]).sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        print(f"Loaded {len(df)} transactions")
        print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features from the dataset.
        """
        feature_df = df.copy()

        # Create synthetic transaction ID
        if 'transaction_id' not in feature_df.columns:
            feature_df['transaction_id'] = [f"CC{i+1:06d}" for i in range(len(feature_df))]

        # Convert Time (seconds) to readable datetime
        if 'Time' in df.columns:
            start_date = datetime(2024, 1, 1)
            feature_df['datetime'] = feature_df['Time'].apply(
                lambda x: start_date + timedelta(seconds=int(x))
            )
            feature_df['hour'] = feature_df['datetime'].dt.hour
            feature_df['day_of_week'] = feature_df['datetime'].dt.dayofweek

        # Amount statistics
        if 'Amount' in df.columns:
            feature_df['amount_zscore'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
            feature_df['is_high_amount'] = feature_df['Amount'] > feature_df['Amount'].quantile(0.95)
            feature_df['is_low_amount'] = feature_df['Amount'] < feature_df['Amount'].quantile(0.05)

        # Store actual fraud labels if present
        if 'Class' in df.columns:
            feature_df['actual_fraud'] = df['Class']

        return feature_df

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare feature matrix for Isolation Forest.
        Uses V1-V28 features, Time, and Amount.
        """
        # Use all V1-V28 features
        v_features = [f'V{i}' for i in range(1, 29)]
        additional_features = ['Time', 'Amount']

        # Combine features
        feature_columns = []
        for col in v_features + additional_features:
            if col in df.columns:
                feature_columns.append(col)

        self.feature_columns = feature_columns
        X = df[feature_columns].fillna(0)

        return X

    def fit_predict(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Fit Isolation Forest and predict anomalies.

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
            n_jobs=-1,
            verbose=0
        )

        print("Training Isolation Forest model...")
        predictions = self.model.fit_predict(X_scaled)
        anomaly_scores = self.model.score_samples(X_scaled)

        # Add predictions to dataframe
        df['anomaly'] = predictions
        df['anomaly_score'] = anomaly_scores
        df['is_anomaly'] = df['anomaly'] == -1

        # Calculate risk score (0-100 scale)
        min_score = anomaly_scores.min()
        max_score = anomaly_scores.max()
        df['risk_score'] = ((1 - (anomaly_scores - min_score) / (max_score - min_score)) * 100).round(2)

        # Calculate performance metrics if actual labels are available
        stats = self._calculate_statistics(df)

        return df, stats

    def _calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate detection statistics and performance metrics."""
        stats = {
            'total_transactions': len(df),
            'anomalies_detected': int(df['is_anomaly'].sum()),
            'anomaly_rate': float(df['is_anomaly'].mean() * 100),
            'high_risk_transactions': int((df['risk_score'] > 75).sum()),
            'features_used': len(self.feature_columns)
        }

        # If ground truth labels are available, calculate accuracy metrics
        if 'actual_fraud' in df.columns:
            from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

            y_true = df['actual_fraud']
            y_pred = df['is_anomaly'].astype(int)

            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            stats.update({
                'actual_frauds': int(y_true.sum()),
                'actual_fraud_rate': float(y_true.mean() * 100),
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn),
                'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                'f1_score': float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
            })

            # Calculate accuracy
            stats['accuracy'] = float((tp + tn) / (tp + tn + fp + fn))

            # ROC AUC using risk scores
            try:
                stats['roc_auc'] = float(roc_auc_score(y_true, df['risk_score']))
            except:
                stats['roc_auc'] = 0.0

        return stats

    def get_top_anomalies(self, df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        """Get top N most anomalous transactions."""
        return df.nlargest(n, 'risk_score')

    def analyze(self, filepath: str, sample_size: int = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete analysis pipeline.

        Args:
            filepath: Path to CSV file
            sample_size: Optional - limit dataset size for faster processing

        Returns:
            Tuple of (analyzed DataFrame, statistics dictionary)
        """
        # Load data
        df = self.load_data(filepath, sample_size)

        # Fit model and predict
        df, stats = self.fit_predict(df)

        return df, stats


if __name__ == "__main__":
    # Example usage
    detector = CreditCardFraudDetector(contamination=0.01)  # Lower contamination for credit card data

    print("Credit Card Fraud Detection System initialized")
    print("Use the Streamlit UI to analyze your data")
