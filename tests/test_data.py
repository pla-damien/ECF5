"""Tests pour le module churnguard.data."""

import pandas as pd

from churnguard.data import load_data, preprocess


def test_load_data_returns_dataframe(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,"
        "PhoneService,MultipleLines,InternetService,OnlineSecurity,"
        "OnlineBackup,DeviceProtection,TechSupport,StreamingTV,"
        "StreamingMovies,Contract,PaperlessBilling,PaymentMethod,"
        "MonthlyCharges,TotalCharges,Churn\n"
        "1234,Male,0,Yes,No,12,Yes,No,Fiber optic,No,No,No,No,No,No,"
        "Month-to-month,Yes,Electronic check,70.0,840.0,No\n"
    )
    result = load_data(str(csv_file))
    assert isinstance(result, pd.DataFrame)


def test_load_data_has_expected_columns(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,"
        "PhoneService,MultipleLines,InternetService,OnlineSecurity,"
        "OnlineBackup,DeviceProtection,TechSupport,StreamingTV,"
        "StreamingMovies,Contract,PaperlessBilling,PaymentMethod,"
        "MonthlyCharges,TotalCharges,Churn\n"
        "1234,Male,0,Yes,No,12,Yes,No,Fiber optic,No,No,No,No,No,No,"
        "Month-to-month,Yes,Electronic check,70.0,840.0,No\n"
    )
    result = load_data(str(csv_file))
    assert len(result.columns) == 21


def test_preprocess_returns_features_and_target():
    df = pd.DataFrame(
        {
            "customerID": ["1234"],
            "gender": ["Male"],
            "SeniorCitizen": [0],
            "Partner": ["Yes"],
            "Dependents": ["No"],
            "tenure": [12],
            "PhoneService": ["Yes"],
            "MultipleLines": ["No"],
            "InternetService": ["Fiber optic"],
            "OnlineSecurity": ["No"],
            "OnlineBackup": ["No"],
            "DeviceProtection": ["No"],
            "TechSupport": ["No"],
            "StreamingTV": ["No"],
            "StreamingMovies": ["No"],
            "Contract": ["Month-to-month"],
            "PaperlessBilling": ["Yes"],
            "PaymentMethod": ["Electronic check"],
            "MonthlyCharges": [70.0],
            "TotalCharges": ["840.0"],
            "Churn": ["No"],
        }
    )
    X, y = preprocess(df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_preprocess_handles_missing_total_charges():
    df = pd.DataFrame(
        {
            "customerID": ["1234", "5678"],
            "gender": ["Male", "Female"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "tenure": [12, 0],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "No"],
            "InternetService": ["Fiber optic", "DSL"],
            "OnlineSecurity": ["No", "No"],
            "OnlineBackup": ["No", "No"],
            "DeviceProtection": ["No", "No"],
            "TechSupport": ["No", "No"],
            "StreamingTV": ["No", "No"],
            "StreamingMovies": ["No", "No"],
            "Contract": ["Month-to-month", "Month-to-month"],
            "PaperlessBilling": ["Yes", "Yes"],
            "PaymentMethod": ["Electronic check", "Electronic check"],
            "MonthlyCharges": [70.0, 20.0],
            "TotalCharges": ["840.0", " "],
            "Churn": ["No", "Yes"],
        }
    )
    X, y = preprocess(df)
    assert len(X) == 1
