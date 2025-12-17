<div align="center">

# 🩺 ML-Heart-Disease-Prediction
*Team-Based UAS Project – Healthcare*

![Python](https://img.shields.io/badge/Python-3.x-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-orange)
![Flask](https://img.shields.io/badge/Flask-Web_App-green)

</div>

---

**#Demo Website Prediksi Penyakit Jantung**
![Image](https://github.com/user-attachments/assets/a4481df6-d299-412b-a88a-5cd99f922277)

## 1. Pendahuluan
Proyek ini bertujuan untuk membangun model Machine Learning (ML) yang mampu memprediksi risiko seseorang menderita penyakit jantung berdasarkan data klinis enam parameter. Proyek ini sepenuhnya diimplementasikan melalui alur kerja Git/GitHub yang terstruktur.

**Tema Machine Learning:** Healthcare (Pelayanan Kesehatan)

## 2. Anggota Kelompok & Bukti Kolaborasi Git Flow

Seluruh pekerjaan dalam proyek ini dilakukan secara kolaboratif menggunakan Git Flow. Setiap anggota berkontribusi melalui branch masing-masing, kemudian digabungkan ke branch utama menggunakan pull request.

| Anggota      | Peran                    | Kontribusi Utama                                                                                 | Branch                                      |
| :----------- | :----------------------- | :----------------------------------------------------------------------------------------------- | :------------------------------------------ |
| *Chalista Nida Nafilla (M0125010)* | Project Lead / Reviewer  | Setup repositori, review dan merge Pull Requests (PRs)                                           | main                                      |
| *Azza Noor Arieva (M0125008)*     | Model Training           | Data preprocessing (scaling), data splitting, model training (Random Forest), dan evaluasi model | feature/model-build                       |
| *Hanifah Putri Solikhah (M0125015)*     | Dokumentasi | Finalisasi dokumentasi (README) dan implementasi web deployment menggunakan Flask/HTML           | feature/hani-documentation|


## 3. Tahapan Pengembangan & Hasil Akhir

### A. Data Preprocessing 
* **Tujuan:** Menyiapkan dan membersihkan data untuk pelatihan model.
* **Script:** `src/data_preprocessing.py`
* **Proses Kunci:** Data dibagi 80% (Training) dan 20% (Testing). Fitur diskalakan menggunakan `StandardScaler` (`src/scaler.joblib`) untuk memastikan fitur memiliki kontribusi yang setara.
* **Output Data:** `data/X_train.csv`, `data/y_train.csv`, dll.

### B. Pembangunan & Pengujian Model 
* **Model:** Random Forest Classifier.
* **Script:** `src/model_training.py`
* **Langkah:** Model dilatih menggunakan data yang sudah di-scale.
* **Model Final:** Disimpan sebagai `src/final_model.joblib`.
* **Pengujian pada Data Testing.**

### C. Implementasi Hasil Akhir (Deployment) 
Untuk menunjukkan model yang dapat dijalankan, kami membuat aplikasi web sederhana:
* **Backend Server:** Flask (`src/app.py`), yang memuat model (`final_model.joblib`) dan *scaler*.
* **Frontend User Interface:** HTML (`templates/index.html`), yang menyediakan *form input* klinis pasien.
* **Cara Menjalankan Lokal:**
    1.  Instal dependensi: `pip install -r requirements.txt` (pastikan Flask, pandas, scikit-learn, joblib ada).
    2.  Jalankan server: `python src/app.py`
    3.  Akses di browser: `http://127.0.0.1:5000/`

## 4. Dependencies (`requirements.txt`)
Daftar library Python yang diperlukan untuk menjalankan proyek.


