import os
import pandas as pd
import requests
from sklearn.preprocessing import RobustScaler


def download_dataset(url, save_path):
    if os.path.exists(save_path):
        print(f"[INFO] File '{save_path}' already exists. Skipping download.")
        return

    print(f"[INFO] Downloading dataset from {url}...")
    response = requests.get(url)
    response.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(response.content)

    print("[INFO] Download complete.")


def preprocess_data(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file '{input_path}' not found.")

    print(f"[INFO] Loading dataset from '{input_path}'...")
    df = pd.read_csv(input_path)

    # 1. Cleaning numeric object columns
    cols_to_float = ['chol_hdl_ratio', 'bmi', 'waist_hip_ratio']
    print("[INFO] Cleaning numeric columns...")
    for col in cols_to_float:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(',', '.', regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"  - Cleaned: {col}")

    # 2. Drop ID & redundant columns
    drop_cols = [
        'patient_number',
        'weight', 'height',
        'waist', 'hip',
        'hdl_chol',
        'diastolic_bp'
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    print("[INFO] Dropped unnecessary and redundant columns.")

    # 3. Handle missing values 
    print("[INFO] Handling missing values...")
    df.fillna(df.median(numeric_only=True), inplace=True)

    # 4. Encoding / Mapping
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    if 'diabetes' in df.columns:
        df['diabetes'] = df['diabetes'].map({'No diabetes': 0, 'Diabetes': 1})

    print("[INFO] Encoded categorical variables.")

    # 5. Scaling numeric features
    print("[INFO] Scaling numeric features...")

    target_col = "diabetes"
    exclude_cols = [target_col, "gender"]

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    scaler = RobustScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print(f"[INFO] Scaled columns: {numeric_cols}")

    # 6. Save preprocessed dataset
    df.to_csv(output_path, index=False)
    print(f"[INFO] Preprocessed data saved to '{output_path}'")

    print("\n[INFO] Dataset info after preprocessing:")
    print(df.info())


if __name__ == "__main__":
    DATASET_URL = (
        "https://raw.githubusercontent.com/"
        "sahrul3114/Eksperimen_SML_Syahrul-Akbar-Ramdhani/main/diabetes.csv"
    )
    RAW_FILE = "diabetes.csv"
    PROCESSED_FILE = "diabetes_preprocessed.csv"

    try:
        download_dataset(DATASET_URL, RAW_FILE)
        preprocess_data(RAW_FILE, PROCESSED_FILE)
    except Exception as e:

        print(f"[ERROR] {e}")
