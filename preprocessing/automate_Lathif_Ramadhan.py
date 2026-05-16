import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- KONFIGURASI ---
INPUT_PATH = "https://raw.githubusercontent.com/LatiefDataVisionary/Eksperimen_SML_Lathif-Ramadhan/refs/heads/main/Telco-Customer-Churn_Raw.csv"
ARTIFACT_DIR = "models/artifacts/"
DATA_CLEAN_DIR = "preprocessing/data_clean/"

def load_data(path):
    """Memuat dataset dari path atau URL."""
    print(f"[INFO] Loading data dari: {path}")
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] Gagal memuat data: {e}")
        return None

def clean_data(df):
    """Membersihkan data mentah."""
    print("[INFO] Membersihkan data...")
    df_clean = df.copy()
    
    # Hapus customerID jika ada
    if 'customerID' in df_clean.columns:
        df_clean.drop(columns=['customerID'], inplace=True)
    
    # Handling TotalCharges
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'].replace(' ', np.nan), errors='coerce')
    median_val = df_clean['TotalCharges'].median()
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(median_val)
    
    return df_clean

def split_data(df, target='Churn'):
    """Membagi data menjadi Train dan Test."""
    print("[INFO] Splitting data (80% Train, 20% Test)...")
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def preprocess_pipeline(X_train, X_test, y_train, y_test):
    """Transformasi data dengan pencegahan Data Leakage."""
    print("[INFO] Memulai preprocessing pipeline...")
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    
    X_train_proc, X_test_proc = X_train.copy(), X_test.copy()

    # 1. Target Encoding
    le_target = LabelEncoder()
    y_train_enc = le_target.fit_transform(y_train)
    y_test_enc = le_target.transform(y_test)
    joblib.dump(le_target, os.path.join(ARTIFACT_DIR, 'target_encoder.joblib'))

    # 2. Feature Preprocessing
    num_cols = X_train_proc.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train_proc.select_dtypes(include=['object']).columns.tolist()

    # Encoding Kategorikal
    for col in cat_cols:
        le_feat = LabelEncoder()
        if X_train_proc[col].nunique() <= 2:
            X_train_proc[col] = le_feat.fit_transform(X_train_proc[col])
            X_test_proc[col] = le_feat.transform(X_test_proc[col])
        else:
            X_train_proc = pd.get_dummies(X_train_proc, columns=[col], drop_first=True)
            X_test_proc = pd.get_dummies(X_test_proc, columns=[col], drop_first=True)
            X_test_proc = X_test_proc.reindex(columns=X_train_proc.columns, fill_value=0)

    # Scaling Numerik (Fit hanya pada Train)
    scaler = StandardScaler()
    X_train_proc[num_cols] = scaler.fit_transform(X_train_proc[num_cols])
    X_test_proc[num_cols] = scaler.transform(X_test_proc[num_cols])
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, 'scaler.joblib'))

    # Merge kembali dengan target untuk output CSV
    train_final = X_train_proc.copy()
    train_final['Churn'] = y_train_enc
    
    test_final = X_test_proc.copy()
    test_final['Churn'] = y_test_enc

    return train_final, test_final

def main():
    # Pastikan folder output ada
    os.makedirs(DATA_CLEAN_DIR, exist_ok=True)

    # Eksekusi Pipeline
    raw_df = load_data(INPUT_PATH)
    if raw_df is not None:
        cleaned_df = clean_data(raw_df)
        X_train, X_test, y_train, y_test = split_data(cleaned_df)
        
        train_csv, test_csv = preprocess_pipeline(X_train, X_test, y_train, y_test)

        # Simpan CSV
        print(f"[INFO] Menyimpan hasil ke: {DATA_CLEAN_DIR}")
        train_csv.to_csv(os.path.join(DATA_CLEAN_DIR, 'train_cleaned.csv'), index=False)
        test_csv.to_csv(os.path.join(DATA_CLEAN_DIR, 'test_cleaned.csv'), index=False)
        
        # Full cleaned (gabungan hasil preprocess)
        pd.concat([train_csv, test_csv]).to_csv(os.path.join(DATA_CLEAN_DIR, 'full_cleaned.csv'), index=False)
        
        print("[SUCCESS] Proses automasi selesai!")

if __name__ == '__main__':
    main()
