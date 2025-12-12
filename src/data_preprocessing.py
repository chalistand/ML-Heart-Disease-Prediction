import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os # Untuk navigasi path file yang lebih aman

# Tentukan PATH relatif ke file data mentah
# Asumsi script ini (src/data_preprocessing.py) dijalankan dari root folder src/
# sehingga kita perlu naik satu level (..) untuk mencapai folder data/
DATA_FILE_PATH = '../data/heart_disease_data.csv'

def load_data(file_path):
    """Memuat data dari file CSV."""
    try:
        # Menangani path relatif
        script_dir = os.path.dirname(__file__) 
        abs_file_path = os.path.join(script_dir, file_path)
        
        df = pd.read_csv(abs_file_path)
        print(f"Data '{os.path.basename(file_path)}' berhasil dimuat.")
        print(f"Jumlah baris awal: {len(df)}")
        return df
    except FileNotFoundError:
        print(f"🚨 Error: File tidak ditemukan di {abs_file_path}")
        print("Pastikan Anda sudah menempatkan 'heart_disease_data.csv' di dalam folder 'data/'.")
        return None

def preprocess_and_split_data(df):
    """
    Melakukan pembersihan data, scaling fitur, dan pembagian data training/testing.
    """
    if df is None:
        return None, None, None, None

    # --- 1. Pembersihan Data (Handling Missing Values) ---
    # Jika ada baris dengan nilai kosong (NaN), kita hapus (drop)
    rows_before = len(df)
    df.dropna(inplace=True)
    rows_after = len(df)
    if rows_before != rows_after:
        print(f"⚠️ Perhatian: {rows_before - rows_after} baris yang memiliki nilai kosong telah dihapus.")

    # --- 2. Pemisahan Fitur (X) dan Target (y) ---
    # 'target' adalah variabel dependen (yang ingin diprediksi)
    if 'target' not in df.columns:
        print("🚨 Error: Kolom 'target' tidak ditemukan di dataset.")
        return None, None, None, None
        
    X = df.drop('target', axis=1)
    y = df['target']
    
    # --- 3. Pembagian Data Training (80%) dan Testing (20%) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 4. Feature Scaling (Normalisasi/Standarisasi) ---
    # Standarisasi diperlukan agar semua fitur memiliki skala yang sama, 
    # ini sangat penting untuk beberapa model ML
    scaler = StandardScaler()
    
    # Fit scaler HANYA pada data training untuk menghindari Data Leakage
    X_train_scaled = scaler.fit_transform(X_train)
    # Terapkan transformasi pada data testing
    X_test_scaled = scaler.transform(X_test)

    # Ubah kembali ke DataFrame agar mudah dibaca
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    print("\nProses Data Preprocessing Selesai.")
    print(f"Ukuran Data Training (80%): {X_train_scaled.shape[0]} baris")
    print(f"Ukuran Data Testing (20%): {X_test_scaled.shape[0]} baris")
    
    # Menyimpan scaler untuk digunakan kembali saat prediksi data baru
    joblib.dump(scaler, '../src/scaler.joblib')
    print("StandardScaler berhasil disimpan di '../src/scaler.joblib'.")

    return X_train_scaled, X_test_scaled, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test):
    """Menyimpan data yang sudah diproses ke folder data/."""
    try:
        X_train.to_csv('../data/X_train.csv', index=False)
        X_test.to_csv('../data/X_test.csv', index=False)
        y_train.to_csv('../data/y_train.csv', index=False, header=True)
        y_test.to_csv('../data/y_test.csv', index=False, header=True)
        print("\nData train/test yang sudah diskalakan berhasil disimpan.")
    except Exception as e:
        print(f"🚨 Error saat menyimpan data: {e}")

if __name__ == '__main__':
    from joblib import dump as joblib_dump, load as joblib_load
    
    # 1. Muat Data
    data_df = load_data(DATA_FILE_PATH)
    
    # 2. Preprocess dan Split
    X_train, X_test, y_train, y_test = preprocess_and_split_data(data_df)
    
    # 3. Simpan Hasil
    if X_train is not None:
        save_processed_data(X_train, X_test, y_train, y_test)
        