import pandas as pd
import numpy as np

def processed_data(df_path):
    df = pd.read_csv(df_path)
    if "id" in df.columns:
        df.drop(["id"], axis=1, inplace=True)

    grade_mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
    df["loan_grade_ord"] = df["loan_grade"].map(grade_mapping)
    df = df.drop(columns="loan_grade")
    
    # Handle Extreme Outliers ages > 100
    df["person_age"] = np.where(df["person_age"] > 100, 100, df["person_age"])
    df["person_emp_length"] = np.where(df["person_emp_length"] > 60, 60, df["person_emp_length"])

    # handle Employment longer than age
    df["person_emp_length"] = np.where(
        df["person_emp_length"] > df["person_age"],
        (df["person_age"] - 16).clip(lower=0),  # Ensures it doesn't go below 0
        df["person_emp_length"],
    )

    return df

def feature_engineer(df):
    # Financial Ratios (High predictive power for loans)
    # df["income_per_year_emp"] = df["person_income"] / (df["person_emp_length"] + 1)
   # Age & Employment Interaction
    # df["age_emp_ratio"] = df["person_emp_length"] / (df["person_age"] + 1)
    # Interest Burden Interaction (Flags high risk tiers)
    df["interest_burden"] = df["loan_int_rate"] * df["loan_percent_income"]
    # Income Residual 
    df["disposable_income_proxy"] = df["person_income"] - df["loan_amnt"]
    # Credit History Velocity
    # df["credit_history_velocity"] = df["cb_person_cred_hist_length"] / (df["person_age"] + 1)
    # Employment Stability Index (Interaction of History and Job Tenure)
    # df["stability_index"] = df["person_emp_length"] * (df["cb_person_cred_hist_length"] + 1)
    df["loan_to_annual_income_ratio"] = (df["loan_amnt"] / df["person_income"])
    return df