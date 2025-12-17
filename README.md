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
Proyek ini bertujuan untuk membangun model *Machine Learning (ML)* yang mampu memprediksi risiko seseorang menderita penyakit jantung berdasarkan data klinis enam parameter. Model melakukan klasifikasi dengan dua kelas, yaitu *Target = 1* (berisiko penyakit jantung) dan *Target = 0* (tidak berisiko penyakit jantung).

Proyek ini dikembangkan sebagai bagian dari *Ujian Akhir Semester (UAS)* dan sepenuhnya diimplementasikan melalui alur kerja *Git/GitHub* yang terstruktur, mencakup penggunaan repository, branch, commit, dan pull request sebagai bukti kolaborasi tim.

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
* *Tujuan:* Menyiapkan dan membersihkan data agar siap digunakan untuk pelatihan model.
* *Script:* src/data_preprocessing.py
* *Proses Kunci:*

  * Pembagian data menjadi *80% data training* dan *20% data testing*
  * Normalisasi fitur menggunakan *StandardScaler* untuk menyamakan skala data
* *Output:*

  * data/X_train.csv
  * data/X_test.csv
  * data/y_train.csv
  * data/y_test.csv
  * src/scaler.joblib

---

### B. Pembangunan & Pengujian Model 
* *Model:* Random Forest Classifier
* *Script:* src/model_training.py
* *Proses:*

  * Pelatihan model menggunakan data yang telah melalui proses scaling
  * Evaluasi model menggunakan data testing
* *Model Final:* src/final_model.joblib

---

### C. Implementasi Hasil Akhir (Deployment) 
Sebagai bentuk implementasi nyata, model yang telah dibangun diintegrasikan ke dalam aplikasi web sederhana.

* *Backend:* Flask (src/app.py)
* *Frontend:* HTML (templates/index.html)
* *Fitur Aplikasi:*

  * Form input data klinis pasien
  * Menampilkan hasil prediksi risiko penyakit jantung

---

## ⚙ Cara Menjalankan Proyek Secara Lokal

Langkah Langkah Menjalankan Proyek Secara Lokal</strong></summary>

<br>

*ML-Heart-Disease-Prediction*

Ikuti langkah-langkah berikut untuk menjalankan aplikasi *Prediksi Penyakit Jantung* pada komputer lokal.

---

### 1️⃣ Pastikan Python Terinstal
```bash
python --version
```

---

### 2️⃣ Clone Repository Proyek
```bash
git clone https://github.com/chalistand/ML-Heart-Disease-Prediction.git
cd ML-Heart-Disease-Prediction
```

---

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

### 4️⃣ Jalankan Aplikasi
```bash
python src/app.py
```

---

### 5️⃣ Akses Aplikasi
```text
http://127.0.0.1:5000/
```

---

### 6️⃣ Menghentikan Aplikasi
Tekan *Ctrl + C* pada terminal untuk menghentikan aplikasi.

✨ *Catatan:*  
Pastikan file final_model.joblib da

---

## 4. Dependencies (requirements.txt)

Library Python yang digunakan dalam proyek ini meliputi:

* Python
* Flask
* Pandas
* NumPy
* Scikit-learn
* Joblib

Seluruh dependensi tercantum lengkap pada file requirements.txt.

---

## 5. Kesimpulan

Proyek *ML-Heart-Disease-Prediction* berhasil mengimplementasikan alur kerja machine learning secara menyeluruh, mulai dari pengolahan data, pelatihan model, evaluasi, hingga deployment dalam bentuk aplikasi web. Selain aspek teknis, proyek ini juga menekankan pentingnya kerja sama tim dan penggunaan GitHub sebagai alat version control dalam pengembangan proyek secara profesional.

---
