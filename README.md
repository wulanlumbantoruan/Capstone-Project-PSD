# Capstone Project: Adult Income Prediction

## 1. Project Overview

**Adult Income Prediction** merupakan proyek klasifikasi untuk memprediksi tingkat pendapatan tahunan seseorang berdasarkan data demografi dan pekerjaan. Prediksi dibagi menjadi dua kelas, yaitu pendapatan **lebih dari 50K USD (>50K)** dan **50K USD atau kurang (<=50K)**.

Proyek ini membandingkan empat algoritma klasifikasi untuk memperoleh model dengan performa terbaik. Model terpilih kemudian digunakan dalam aplikasi prediksi berbasis Streamlit.

## 2. Dataset

Dataset yang digunakan adalah **Adult Census Income Dataset** dari Kaggle.

- **Source:** [Adult Census Income Dataset](https://www.kaggle.com/datasets/uciml/adult-census-income)
- **Original data:** 32.561 baris
- **Jumlah fitur:** 14
- **Jumlah kelas:** 2
- **Data setelah preprocessing:** 32.537 baris
- **Fitur yang digunakan model:** 6

Dataset ini berisi data sensus penduduk Amerika Serikat tahun 1994. Atribut yang tersedia mencakup informasi demografi dan pekerjaan yang digunakan untuk memprediksi kategori pendapatan.

### Data Preprocessing

Tahapan preprocessing meliputi:
- Menghapus data duplikat.
- Menyiapkan fitur yang digunakan dalam pemodelan.
- Membagi data menjadi data latih dan data uji dengan rasio 80:20.

Pembagian data menghasilkan:
- Data latih: 26.029 baris.
- Data uji: 6.508 baris.

## 3. Classification Algorithms

Empat algoritma klasifikasi digunakan dan dibandingkan dalam proyek ini:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Multi-Layer Perceptron (MLP)

Setiap algoritma melalui proses *hyperparameter tuning* untuk mendapatkan konfigurasi model yang optimal.

Berdasarkan hasil evaluasi, **Random Forest** dipilih sebagai model terbaik untuk digunakan dalam aplikasi prediksi.

## 4. Model Evaluation

Evaluasi model dilakukan menggunakan beberapa metrik klasifikasi, dengan hasil model Random Forest sebagai berikut:

| Evaluation Metric | Score |
|---|---:|
| AUC-ROC | 0.903 |
| Accuracy | 0.818 |
| Recall (>50K) | 0.822 |
| F1-score (>50K) | 0.685 |

Evaluasi dilakukan pada data uji menggunakan threshold optimal sebesar **0.304**.

Perbandingan performa keempat algoritma dapat dilihat pada halaman Model Evaluation dalam aplikasi.

## 5. Application

Model Random Forest yang telah dilatih digunakan dalam aplikasi berbasis **Streamlit** untuk melakukan prediksi kategori pendapatan berdasarkan input pengguna.

Aplikasi menyediakan beberapa bagian utama:

- **Dashboard:** Menampilkan ringkasan proyek dan informasi performa model.
- **Prediction:** Melakukan prediksi kategori pendapatan berdasarkan input fitur.
- **Model Evaluation:** Menampilkan hasil evaluasi dan perbandingan algoritma.
- **Dataset Information:** Menampilkan informasi dan karakteristik dataset.

## 6. Tools and Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Jupyter Notebook
- Git & GitHub

## 7. Repository

**GitHub:** [Capstone-Project-PSD](https://github.com/wulanlumbantoruan/Capstone-Project-PSD)

**Dataset:** [Adult Census Income Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/adult-census-income)
