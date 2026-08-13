"""
Generates synthetic Telco Customer Churn dataset matching Kaggle standard schema.
"""

import os

import numpy as np
import pandas as pd


def generate_telco_churn_data(num_samples: int = 7043, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    
    customer_ids = [f"{np.random.randint(1000, 9999)}-{chr(65 + i % 26)}{chr(65 + (i * 3) % 26)}{chr(65 + (i * 7) % 26)}" for i in range(num_samples)]
    
    genders = np.random.choice(["Male", "Female"], size=num_samples)
    senior_citizens = np.random.choice([0, 1], size=num_samples, p=[0.84, 0.16])
    partners = np.random.choice(["Yes", "No"], size=num_samples, p=[0.48, 0.52])
    dependents = np.random.choice(["Yes", "No"], size=num_samples, p=[0.30, 0.70])
    
    # Tenure in months (0 - 72)
    tenure = np.random.randint(0, 73, size=num_samples)
    
    phone_service = np.random.choice(["Yes", "No"], size=num_samples, p=[0.90, 0.10])
    multiple_lines = []
    for ps in phone_service:
        if ps == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.45, 0.55]))
            
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=num_samples, p=[0.34, 0.44, 0.22])
    
    def get_internet_addon(net_service):
        res = []
        for iserv in net_service:
            if iserv == "No":
                res.append("No internet service")
            else:
                res.append(np.random.choice(["Yes", "No"], p=[0.40, 0.60]))
        return res

    online_security = get_internet_addon(internet_service)
    online_backup = get_internet_addon(internet_service)
    device_protection = get_internet_addon(internet_service)
    tech_support = get_internet_addon(internet_service)
    streaming_tv = get_internet_addon(internet_service)
    streaming_movies = get_internet_addon(internet_service)

    contracts = np.random.choice(["Month-to-month", "One year", "Two year"], size=num_samples, p=[0.55, 0.21, 0.24])
    paperless_billing = np.random.choice(["Yes", "No"], size=num_samples, p=[0.59, 0.41])
    payment_methods = np.random.choice([
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ], size=num_samples, p=[0.34, 0.23, 0.22, 0.21])

    # Monthly charges depend realistic on features
    base_charge = 20.0
    monthly_charges = (
        base_charge
        + (internet_service == "Fiber optic") * 45.0
        + (internet_service == "DSL") * 25.0
        + (np.array(streaming_tv) == "Yes") * 10.0
        + (np.array(streaming_movies) == "Yes") * 10.0
        + (np.array(tech_support) == "Yes") * 5.0
        + (np.array(online_security) == "Yes") * 5.0
        + np.random.normal(0, 3, size=num_samples)
    )
    monthly_charges = np.clip(monthly_charges, 18.25, 118.75).round(2)
    
    total_charges = (monthly_charges * tenure + np.random.normal(0, 10, size=num_samples)).round(2)
    total_charges = np.where(tenure == 0, 0.0, total_charges)
    total_charges = np.clip(total_charges, 0, None)

    # Realistic churn probability calculation
    churn_score = (
        (contracts == "Month-to-month") * 1.5
        + (internet_service == "Fiber optic") * 1.0
        + (np.array(tech_support) == "No") * 0.8
        + (np.array(online_security) == "No") * 0.8
        + (payment_methods == "Electronic check") * 0.6
        - (tenure / 12.0) * 0.4
        - (contracts == "Two year") * 1.2
        + np.random.normal(0, 0.8, size=num_samples)
    )

    churn_prob = 1 / (1 + np.exp(-churn_score))
    churn = np.where(churn_prob > 0.62, "Yes", "No")

    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": genders,
        "SeniorCitizen": senior_citizens,
        "Partner": partners,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contracts,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_methods,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn
    })

    return df


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    data_path = "data/raw/telco_churn.csv"
    df = generate_telco_churn_data()
    df.to_csv(data_path, index=False)
    print(f"Generated synthetic dataset with {len(df)} rows at {data_path}")
