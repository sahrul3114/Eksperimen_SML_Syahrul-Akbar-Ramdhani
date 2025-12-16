import os
import pandas as pd
import requests


def download_dataset(url, save_path):
    """Download dataset jika belum tersedia."""
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
    """Preprocessing dataset diabetes."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file '{input_path}' not found.")

    print(f"[INFO] Loading dataset from '{input_path}'...")
    df = pd.read_csv(input_path)

    # 1. Cleaning kolom numerik bertipe object
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

    # 2. Drop kolom ID & variabel redundant (multikolinearitas)
    drop_cols = [
        'patient_number',
        'weight', 'height',
        'waist', 'hip',
        'hdl_chol',
        'diastolic_bp'
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    print("[INFO] Dropped unnecessary and redundant columns.")

    # 3. Encoding variabel kategorik
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    if 'diabetes' in df.columns:
        df['diabetes'] = df['diabetes'].map({'No diabetes': 0, 'Diabetes': 1})

    print("[INFO] Encoded categorical variables.")

    # 4. Simpan hasil preprocessing
    df.to_csv(output_path, index=False)
    print(f"[INFO] Preprocessed data saved to '{output_path}'.")

    # Info ringkas
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