"""
Sample Data Generator
Creates synthetic financial transaction data for testing the fraud detection system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_sample_transactions(n_transactions=1000, fraud_rate=0.1):
    """
    Generate sample financial transaction data with synthetic anomalies.

    Args:
        n_transactions: Number of transactions to generate
        fraud_rate: Proportion of fraudulent/anomalous transactions

    Returns:
        DataFrame containing sample transaction data
    """
    np.random.seed(42)
    random.seed(42)

    # Generate base data
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=random.randint(0, 365)) for _ in range(n_transactions)]

    merchants = [
        'ABC Electronics', 'XYZ Grocery', 'Tech Solutions Inc',
        'Office Supplies Co', 'Restaurant Group', 'Retail Store',
        'Online Marketplace', 'Service Provider', 'Manufacturing Ltd',
        'Consulting Firm', 'Local Hardware', 'Auto Parts Shop'
    ]

    provinces = {
        'ON': 13,  # Ontario HST
        'NS': 15,  # Nova Scotia HST
        'NB': 15,  # New Brunswick HST
        'NL': 15,  # Newfoundland HST
        'PE': 15,  # PEI HST
        'QC': 5,   # Quebec GST (QST separate)
        'BC': 5,   # BC GST
        'AB': 5,   # Alberta GST
        'SK': 5,   # Saskatchewan GST
        'MB': 5    # Manitoba GST
    }

    transactions = []

    for i in range(n_transactions):
        transaction_id = f"TXN{i+1:06d}"
        date = dates[i]
        merchant = random.choice(merchants)
        province = random.choice(list(provinces.keys()))
        expected_tax_rate = provinces[province]

        # Generate normal transaction amounts
        subtotal = round(random.uniform(10, 5000), 2)

        # Determine if this should be an anomaly
        is_fraud = random.random() < fraud_rate

        if is_fraud:
            # Create different types of anomalies
            anomaly_type = random.choice(['wrong_tax', 'duplicate', 'unusual_amount'])

            if anomaly_type == 'wrong_tax':
                # Wrong tax rate applied
                wrong_rate = random.choice([0, 3, 8, 20, 25])
                tax = round(subtotal * wrong_rate / 100, 2)
            elif anomaly_type == 'unusual_amount':
                # Unusual transaction amount
                subtotal = round(random.uniform(10000, 50000), 2)
                tax = round(subtotal * expected_tax_rate / 100, 2)
            else:
                # Normal for now (will duplicate later)
                tax = round(subtotal * expected_tax_rate / 100, 2)
        else:
            # Normal transaction with correct tax
            tax = round(subtotal * expected_tax_rate / 100, 2)

        total = round(subtotal + tax, 2)

        transactions.append({
            'transaction_id': transaction_id,
            'date': date.strftime('%Y-%m-%d %H:%M:%S'),
            'merchant': merchant,
            'province': province,
            'subtotal': subtotal,
            'tax': tax,
            'total': total,
            'category': random.choice(['Electronics', 'Food', 'Services', 'Supplies', 'Other'])
        })

    df = pd.DataFrame(transactions)

    # Add some duplicate transactions
    n_duplicates = int(n_transactions * 0.05)  # 5% duplicates
    duplicate_indices = random.sample(range(len(df)), n_duplicates)

    for idx in duplicate_indices:
        duplicate_row = df.iloc[idx].copy()
        # Change transaction_id but keep other details the same
        duplicate_row['transaction_id'] = f"TXN{len(df)+1:06d}"
        # Slightly modify date
        original_date = datetime.strptime(duplicate_row['date'], '%Y-%m-%d %H:%M:%S')
        duplicate_row['date'] = (original_date + timedelta(hours=random.randint(1, 48))).strftime('%Y-%m-%d %H:%M:%S')

        df = pd.concat([df, duplicate_row.to_frame().T], ignore_index=True)

    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def main():
    """Generate and save sample data."""
    print("Generating sample financial transaction data...")

    # Generate data
    df = generate_sample_transactions(n_transactions=1000, fraud_rate=0.1)

    # Save to CSV
    output_file = "sample_transactions.csv"
    df.to_csv(output_file, index=False)

    print(f"✅ Generated {len(df)} transactions")
    print(f"📁 Saved to: {output_file}")
    print("\nDataset summary:")
    print(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  - Merchants: {df['merchant'].nunique()}")
    print(f"  - Total amount: ${df['total'].sum():,.2f}")
    print(f"  - Avg transaction: ${df['total'].mean():,.2f}")

    # Show sample
    print("\nFirst 5 transactions:")
    print(df.head().to_string())


if __name__ == "__main__":
    main()
