import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==========================================
# STEP 1: DATA COLLECTION (Simulated Dataset)
# ==========================================
def collect_data():
  """Simulate data collection from a database or CSV file."""
  data = {
      "Age": [25, 30, np.nan, 45, 22, 38],
      "Income": [50000, 60000, 52000, np.nan, 30000, 85000],
      "Hours_Per_Week": [40, 50, 35, 45, 20, 60],
      "City": ["New York", "London", "Paris", "New York", np.nan, "London"],
      "Target_Purchased": [0, 1, 0, 1, 0, 1],
  }
  df = pd.DataFrame(data)
  print("--- Step 1: Raw Data Collected ---")
  print(df)
  return df


# ==========================================
# STEP 2: DATA VALIDATION
# ==========================================
def validate_data(df):
  """Validate basic requirements before processing pipeline."""
  required_columns = ["Age", "Income", "Hours_Per_Week", "City"]

  # Check missing mandatory columns
  for col in required_columns:
    assert col in df.columns, f"Missing required column: {col}"

  # Check value bounds/validations
  assert (df["Age"].dropna() > 0).all(), "Validation Error: Age must be > 0"
  print("\n--- Step 2: Data Validation Passed ---")


# ==========================================
# STEP 3: FEATURE ENGINEERING
# ==========================================
def engineer_features(df):
  """Create derived features to improve predictive quality."""
  df_fe = df.copy()
  # Derived Feature: Hourly Income Estimate (assuming 52 weeks/year)
  df_fe["Hourly_Income"] = df_fe["Income"] / (df_fe["Hours_Per_Week"] * 52)
  print("\n--- Step 3: Feature Engineering Completed ---")
  print(df_fe[["Income", "Hours_Per_Week", "Hourly_Income"]].head())
  return df_fe


# ==========================================
# STEP 4: PIPELINE DESIGN & AUTOMATION
# ==========================================
def create_data_pipeline(numeric_features, categorical_features):
  """Build an automated preprocessing pipeline."""
  # Pipeline for Numeric Features: Impute missing values -> Scale features
  numeric_transformer = Pipeline(
      steps=[
          ("imputer", SimpleImputer(strategy="median")),
          ("scaler", StandardScaler()),
      ]
  )

  # Pipeline for Categorical Features: Impute missing values -> One-Hot Encode
  categorical_transformer = Pipeline(
      steps=[
          (
              "imputer",
              SimpleImputer(strategy="most_frequent"),
          ),  # Mode imputation
          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
      ]
  )

  # Combine pipelines using ColumnTransformer
  preprocessor = ColumnTransformer(
      transformers=[
          ("num", numeric_transformer, numeric_features),
          ("cat", categorical_transformer, categorical_features),
      ]
  )

  return preprocessor


# ==========================================
# STEP 5: EXECUTION OF END-TO-END DATA PIPELINE
# ==========================================
def run_end_to_end_pipeline():
  # 1. Collect
  df = collect_data()

  # 2. Validate
  validate_data(df)

  # 3. Feature Engineering
  df_fe = engineer_features(df)

  # Separate Features and Target
  X = df_fe.drop(columns=["Target_Purchased"])
  y = df_fe["Target_Purchased"]

  # Define Feature Groups
  numeric_features = ["Age", "Income", "Hours_Per_Week", "Hourly_Income"]
  categorical_features = ["City"]

  # Build Pipeline
  pipeline = create_data_pipeline(numeric_features, categorical_features)

  # Train-Test Split
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.33, random_state=42
  )

  # Fit & Transform Training Data
  X_train_processed = pipeline.fit_transform(X_train)
  X_test_processed = pipeline.transform(X_test)

  print("\n--- Step 5: End-to-End Pipeline Execution Successful ---")
  print("Processed Training Shape:", X_train_processed.shape)
  print("Processed Sample Vector:\n", X_train_processed[0])


if __name__ == "__main__":
  run_end_to_end_pipeline()