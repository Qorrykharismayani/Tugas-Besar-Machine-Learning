# Telco Customer Churn Prediction
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

Proyek ini merupakan pemenuhan tugas besar mata kuliah Dasar Ilmu Data yang bertujuan untuk membangun aplikasi machine learning berbasis web untuk melakukan klasifikasi pelanggan terhadap dataset **Telco Customer Churn**.

## Deskripsi Proyek
Tujuan utama dari proyek ini adalah memprediksi pelanggan mana yang berpotensi melakukan *churn* (berhenti berlangganan) dari layanan telekomunikasi. Analisis ini sangat krusial bagi perusahaan dalam merencanakan strategi retensi pelanggan.

Dalam proyek ini, kami melakukan:
- **Pra-pemrosesan Data**: Pembersihan dan penanganan data yang hilang (*missing values*).
- **Seleksi Fitur**: Memilih variabel paling relevan untuk meningkatkan akurasi model.
- **Pemodelan**: Implementasi dan komparasi 4 metode machine learning untuk mendapatkan performa terbaik.
- **Optimasi Hyperparameter**: Melakukan *tuning* pada model untuk memaksimalkan hasil prediksi.
- **Deployment**: Integrasi model ke dalam aplikasi berbasis web untuk kemudahan penggunaan.

## Model Machine Learning
Dalam proyek ini, kami membandingkan performa dari 4 algoritma klasifikasi:
* **Random Forest (RF)**
* **k-Nearest Neighbors (k-NN)**
* **Support Vector Machine (SVM)**
* **Decision Tree (DT)**

## Teknologi yang Digunakan
* **Bahasa**: Python
* **Lingkungan Kerja**: Jupyter Notebook
* **Library ML**: Scikit-learn, Pandas, NumPy
* **Web Framework**: Streamlit
* **Dataset**: [WA_Fn-UseC_-Telco-Customer-Churn.csv](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
