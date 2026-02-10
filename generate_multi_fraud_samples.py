"""
Multi-Type Fraud Sample Data Generator
Generates sample datasets for different fraud types
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_tax_filing_data(n_records=1000, fraud_rate=0.15):
    """Generate tax filing data with evasion patterns."""
    np.random.seed(42)
    random.seed(42)

    data = []
    for i in range(n_records):
        # Normal income distribution
        income = round(np.random.lognormal(11, 0.8), 2)  # Mean around $60k
        reported_income = income

        # Deductions (normally 10-30% of income)
        normal_deduction_rate = random.uniform(0.10, 0.30)
        deductions = round(income * normal_deduction_rate, 2)

        # Business expenses (normally 0-20% of income)
        business_expenses = round(income * random.uniform(0, 0.20), 2)

        # Calculate tax
        if income <= 50000:
            tax_rate = 0.15
        elif income <= 100000:
            tax_rate = 0.205
        elif income <= 150000:
            tax_rate = 0.26
        else:
            tax_rate = 0.29

        taxable_income = max(0, income - deductions)
        tax_paid = round(taxable_income * tax_rate, 2)

        is_fraud = random.random() < fraud_rate

        if is_fraud:
            fraud_type = random.choice(['underreport', 'excess_deduct', 'fake_expenses', 'tax_dodge'])

            if fraud_type == 'underreport':
                # Underreport income by 10-30%
                reported_income = round(income * random.uniform(0.70, 0.90), 2)

            elif fraud_type == 'excess_deduct':
                # Excessive deductions (40-70% of income)
                deductions = round(income * random.uniform(0.40, 0.70), 2)

            elif fraud_type == 'fake_expenses':
                # Unrealistic business expenses
                business_expenses = round(income * random.uniform(0.50, 0.90), 2)

            elif fraud_type == 'tax_dodge':
                # Pay much less tax than expected
                tax_paid = round(tax_paid * random.uniform(0.30, 0.70), 2)

            # Round numbers (common in fabricated data)
            if random.random() < 0.6:
                income = round(income / 1000) * 1000
                deductions = round(deductions / 100) * 100

        data.append({
            'taxpayer_id': f'TP{i+1:06d}',
            'income': income,
            'reported_income': reported_income,
            'deductions': deductions,
            'business_expenses': business_expenses,
            'taxable_income': max(0, reported_income - deductions),
            'tax_paid': tax_paid,
            'credits': round(random.uniform(0, 2000), 2),
            'year': random.choice([2022, 2023, 2024]),
            'is_fraud': int(is_fraud)
        })

    return pd.DataFrame(data)


def generate_invoice_data(n_records=1000, fraud_rate=0.12):
    """Generate invoice/procurement data with fraud patterns."""
    np.random.seed(42)
    random.seed(42)

    vendors = [
        'Acme Corp', 'Global Supplies Inc', 'Tech Solutions Ltd', 'Office Pro',
        'BuildMart', 'Industrial Parts Co', 'Services Unlimited', 'Metro Supplies',
        'Quality Goods Ltd', 'Professional Services Inc'
    ]

    data = []
    invoice_num = 1000

    for i in range(n_records):
        invoice_num += random.randint(1, 5)  # Occasional gaps
        vendor = random.choice(vendors)
        date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 365))

        # Normal invoice amount
        amount = round(np.random.lognormal(6, 1.5), 2)  # Mean around $400

        is_fraud = random.random() < fraud_rate

        if is_fraud:
            fraud_type = random.choice(['split', 'duplicate', 'concentration', 'round'])

            if fraud_type == 'split':
                # Just below approval thresholds
                threshold = random.choice([1000, 5000, 10000])
                amount = round(threshold * random.uniform(0.92, 0.99), 2)

            elif fraud_type == 'duplicate':
                # Duplicate previous invoice
                if len(data) > 5:
                    prev = random.choice(data[-5:])
                    amount = prev['amount']
                    vendor = prev['vendor']

            elif fraud_type == 'concentration':
                # Favor specific vendors (will show in analysis)
                vendor = random.choice(vendors[:2])  # Concentrate on first 2 vendors

            elif fraud_type == 'round':
                # Suspicious round numbers
                amount = round(amount / 100) * 100

        po_number = f'PO{random.randint(1000, 9999)}' if random.random() < 0.8 else ''

        data.append({
            'invoice_number': f'INV{invoice_num}',
            'vendor': vendor,
            'amount': amount,
            'date': date.strftime('%Y-%m-%d'),
            'po_number': po_number,
            'category': random.choice(['Supplies', 'Services', 'Equipment', 'Maintenance']),
            'department': random.choice(['IT', 'Operations', 'Admin', 'Sales']),
            'is_fraud': int(is_fraud)
        })

    return pd.DataFrame(data)


def generate_expense_data(n_records=1000, fraud_rate=0.10):
    """Generate employee expense data with fraud patterns."""
    np.random.seed(42)
    random.seed(42)

    employees = [f'Employee_{chr(65+i)}' for i in range(20)]
    merchants = [
        'Restaurant ABC', 'Hotel XYZ', 'Gas Station', 'Office Depot',
        'Coffee Shop', 'Airline', 'Taxi Service', 'Conference Center'
    ]

    data = []

    for i in range(n_records):
        employee = random.choice(employees)
        merchant = random.choice(merchants)
        date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 365))

        # Normal expense
        amount = round(np.random.lognormal(3.5, 1), 2)  # Mean around $33

        is_fraud = random.random() < fraud_rate

        if is_fraud:
            fraud_type = random.choice(['duplicate', 'just_below', 'excessive', 'round'])

            if fraud_type == 'duplicate':
                # Duplicate receipt
                if len(data) > 5:
                    prev = random.choice([d for d in data if d['employee'] == employee][-3:] if any(d['employee'] == employee for d in data) else data[-3:])
                    amount = prev['amount']
                    merchant = prev['merchant']
                    # Same or next day
                    prev_date = datetime.strptime(prev['date'], '%Y-%m-%d')
                    date = prev_date + timedelta(days=random.randint(0, 1))

            elif fraud_type == 'just_below':
                # Just below limits (50, 100, 200)
                limit = random.choice([50, 100, 200, 500])
                amount = round(limit * random.uniform(0.92, 0.99), 2)

            elif fraud_type == 'excessive':
                # Unusually high for employee
                amount = round(amount * random.uniform(5, 15), 2)

            elif fraud_type == 'round':
                # Suspicious round numbers
                amount = round(amount / 10) * 10

        data.append({
            'expense_id': f'EXP{i+1:06d}',
            'employee': employee,
            'amount': amount,
            'merchant': merchant,
            'date': date.strftime('%Y-%m-%d'),
            'category': random.choice(['Meals', 'Travel', 'Supplies', 'Entertainment']),
            'description': random.choice(['Business meeting', 'Client dinner', 'Office supplies', 'Conference']),
            'is_fraud': int(is_fraud)
        })

    return pd.DataFrame(data)


def generate_gst_hst_data(n_records=1000, fraud_rate=0.08):
    """Generate Canadian GST/HST transaction data."""
    np.random.seed(42)
    random.seed(42)

    provinces = {
        'ON': 13, 'NS': 15, 'NB': 15, 'NL': 15, 'PE': 15,
        'QC': 5, 'BC': 5, 'AB': 5, 'SK': 5, 'MB': 5
    }

    merchants = [
        'Retail Store A', 'Restaurant B', 'Service Provider C',
        'Grocery Store D', 'Electronics E', 'Clothing F'
    ]

    data = []

    for i in range(n_records):
        merchant = random.choice(merchants)
        province = random.choice(list(provinces.keys()))
        expected_tax_rate = provinces[province]
        date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 365))

        # Normal transaction
        subtotal = round(np.random.lognormal(4, 1.2), 2)  # Mean around $55
        tax = round(subtotal * expected_tax_rate / 100, 2)
        total = round(subtotal + tax, 2)

        is_fraud = random.random() < fraud_rate

        if is_fraud:
            fraud_type = random.choice(['wrong_rate', 'duplicate', 'no_tax', 'excess_tax'])

            if fraud_type == 'wrong_rate':
                # Wrong tax rate
                wrong_rate = random.choice([0, 3, 8, 10, 20])
                tax = round(subtotal * wrong_rate / 100, 2)

            elif fraud_type == 'duplicate':
                # Duplicate transaction
                if len(data) > 5:
                    prev = random.choice(data[-5:])
                    subtotal = prev['subtotal']
                    tax = prev['tax']
                    merchant = prev['merchant']

            elif fraud_type == 'no_tax':
                tax = 0

            elif fraud_type == 'excess_tax':
                # Overcharging tax
                tax = round(subtotal * random.uniform(0.18, 0.25), 2)

            total = round(subtotal + tax, 2)

        data.append({
            'transaction_id': f'TXN{i+1:06d}',
            'merchant': merchant,
            'province': province,
            'subtotal': subtotal,
            'tax': tax,
            'total': total,
            'date': date.strftime('%Y-%m-%d'),
            'category': random.choice(['Retail', 'Food', 'Services', 'Other']),
            'is_fraud': int(is_fraud)
        })

    return pd.DataFrame(data)


def main():
    """Generate all sample datasets."""
    print("Generating Multi-Type Fraud Sample Data...\n")

    # Generate datasets
    print("1. Generating Tax Filing Data...")
    tax_df = generate_tax_filing_data(n_records=1000, fraud_rate=0.15)
    tax_df.to_csv('sample_tax_filing.csv', index=False)
    print(f"   [OK] Saved sample_tax_filing.csv ({len(tax_df)} records, {tax_df['is_fraud'].sum()} frauds)")

    print("\n2. Generating Invoice Data...")
    invoice_df = generate_invoice_data(n_records=1000, fraud_rate=0.12)
    invoice_df.to_csv('sample_invoices.csv', index=False)
    print(f"   [OK] Saved sample_invoices.csv ({len(invoice_df)} records, {invoice_df['is_fraud'].sum()} frauds)")

    print("\n3. Generating Expense Data...")
    expense_df = generate_expense_data(n_records=1000, fraud_rate=0.10)
    expense_df.to_csv('sample_expenses.csv', index=False)
    print(f"   [OK] Saved sample_expenses.csv ({len(expense_df)} records, {expense_df['is_fraud'].sum()} frauds)")

    print("\n4. Generating GST/HST Data...")
    gst_df = generate_gst_hst_data(n_records=1000, fraud_rate=0.08)
    gst_df.to_csv('sample_gst_hst.csv', index=False)
    print(f"   [OK] Saved sample_gst_hst.csv ({len(gst_df)} records, {gst_df['is_fraud'].sum()} frauds)")

    print("\n" + "="*60)
    print("[SUCCESS] All sample datasets generated successfully!")
    print("="*60)

    print("\nDataset Summary:")
    print(f"  • Tax Filing:  {len(tax_df):,} records ({tax_df['is_fraud'].mean()*100:.1f}% fraud)")
    print(f"  • Invoices:    {len(invoice_df):,} records ({invoice_df['is_fraud'].mean()*100:.1f}% fraud)")
    print(f"  • Expenses:    {len(expense_df):,} records ({expense_df['is_fraud'].mean()*100:.1f}% fraud)")
    print(f"  • GST/HST:     {len(gst_df):,} records ({gst_df['is_fraud'].mean()*100:.1f}% fraud)")

    print("\nUse these files with the unified fraud detection system!")


if __name__ == "__main__":
    main()
