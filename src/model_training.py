import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib 
import os 
import warnings
warnings.filterwarnings('ignore')

# Tentukan PATH relatif ke root folder (di luar src/)
DATA_DIR = '../data/'
MODEL_PATH = '../src/final_model.joblib'

def get_absolute_path(relative_path):
    """Mendapatkan path absolut."""
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    return os.path.join(script_dir, relative_path)

def load_processed_data():
    """Memuat data training dan testing yang sudah di-scale."""
    try:
        X_train = pd.read_csv(get_absolute_path(DATA_DIR + 'X_train.csv'))
        X_test = pd.read_csv(get_absolute_path(DATA_DIR + 'X_test.csv'))
        y_train = pd.read_csv(get_absolute_path(DATA_DIR + 'y_train.csv')).squeeze()
        y_test = pd.read_csv(get_absolute_path(DATA_DIR + 'y_test.csv')).squeeze()
        
        print("Data train/test berhasil dimuat dan siap training.")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        print("🚨 Error: File data train/test (X_train.csv, dll.) tidak ditemukan.")
        print("Pastikan Anda sudah menjalankan 'python src/data_preprocessing.py' terlebih dahulu.")
        return None, None, None, None

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """Melatih model Random Forest, menguji, dan menyimpan hasilnya."""
    
    if X_train is None:
        return None

    # 1. Inisialisasi Model
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)

    # 2. Melatih Model
    print("\n--- Mulai Melatih Model Random Forest ---")
    model.fit(X_train, y_train)
    print("Pelatihan model selesai.")

    # 3. Menguji Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Tidak Sakit (0)', 'Sakit Jantung (1)'])

    print("\n--- Hasil Evaluasi Model ---")
    print(f"Akurasi Model pada data testing: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(report)
    
    # 4. Menyimpan Model
    joblib.dump(model, get_absolute_path(MODEL_PATH))
    print(f"\n✅ Model Random Forest berhasil disimpan di: {get_absolute_path(MODEL_PATH)}")
    return model

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_processed_data()
    train_and_evaluate_model(X_train, X_test, y_train, y_test)
    