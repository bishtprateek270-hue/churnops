"""
Generates synthetic Telco Customer Churn dataset matching Kaggle standard schema.
"""

import os

import numpy as np
import pandas as pd


def generate_telco_churn_data(num_samples: int = 7043, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    customer_ids = [
        f"{np.random.randint(1000, 9999)}-{chr(65 + i % 26)}{chr(65 + (i * 3) % 26)}{chr(65 + (i * 7) % 26)}"
        for i in range(num_samples)
    ]

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
    payment_methods = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=num_samples,
        p=[0.34, 0.23, 0.22, 0.21],
    )

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
    churn = np.where(churn_prob > np.percentile(churn_prob, 73), "Yes", "No")

    df = pd.DataFrame(
        {
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
            "Churn": churn,
        }
    )

    return df


def generate_kaggle_house_prices_data(num_samples: int = 1460, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    ids = range(1, num_samples + 1)
    overall_qual = np.random.randint(1, 11, size=num_samples)
    gr_liv_area = np.random.randint(600, 4500, size=num_samples)
    total_bsmt_sf = np.random.randint(0, 3000, size=num_samples)
    garage_cars = np.random.randint(0, 4, size=num_samples)
    year_built = np.random.randint(1900, 2021, size=num_samples)
    neighborhood = np.random.choice(["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst"], size=num_samples)

    sale_price = (
        10000.0
        + overall_qual * 15000.0
        + gr_liv_area * 60.0
        + total_bsmt_sf * 40.0
        + garage_cars * 8000.0
        + (year_built - 1900) * 300.0
        + np.random.normal(0, 15000, size=num_samples)
    ).round(2)

    df = pd.DataFrame(
        {
            "Id": ids,
            "OverallQual": overall_qual,
            "GrLivArea": gr_liv_area,
            "TotalBsmtSF": total_bsmt_sf,
            "GarageCars": garage_cars,
            "YearBuilt": year_built,
            "Neighborhood": neighborhood,
            "SalePrice": np.clip(sale_price, 30000.0, None),
        }
    )
    return df


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    telco_path = "data/raw/telco_churn.csv"
    df_telco = generate_telco_churn_data()
    df_telco.to_csv(telco_path, index=False)
    print(f"Generated synthetic dataset with {len(df_telco)} rows at {telco_path}")

    house_path = "data/raw/kaggle_house_prices.csv"
    df_house = generate_kaggle_house_prices_data()
    df_house.to_csv(house_path, index=False)
    print(f"Generated synthetic dataset with {len(df_house)} rows at {house_path}")
