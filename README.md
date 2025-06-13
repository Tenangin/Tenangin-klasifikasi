# Sistem Klasifikasi Gangguan Mental Berbasis Teks

Proyek ini mengembangkan sistem klasifikasi gangguan mental menggunakan teknik Natural Language Processing (NLP) pada data teks. Tujuannya adalah untuk mengidentifikasi dan mengkategorikan jenis gangguan mental dari input teks, seperti postingan media sosial atau transkrip percakapan, dengan menggunakan berbagai model pembelajaran mesin.

## Daftar Isi

- [Gambaran Umum Proyek](#gambaran-umum-proyek)
- [Sumber Data](#sumber-data)
- [Pra-pemrosesan Data](#pra-pemrosesan-data)
- [Analisis Data Eksplorasi (EDA)](#analisis-data-eksplorasi-eda)
- [Model Klasifikasi](#model-klasifikasi)
  - [Model Berbasis Scikit-learn (Logistic Regression, RandomForest, Naive Bayes)](#model-berbasis-scikit-learn-logistic-regression-randomforest-naive-bayes)
  - [Model XGBoost](#model-xgboost)
  - [Model TensorFlow/Keras](#model-tensorflowkeras)
  - [Model TensorFlow dengan 14 Fitur Teratas](#model-tensorflow-dengan-14-fitur-teratas)
- [Cara Menggunakan (Pengaturan Lokal)](#cara-menggunakan-pengaturan-lokal)
- [Inferensi](#Inferensi)

## Gambaran Umum Proyek

Proyek ini berfokus pada pengembangan sistem yang dapat mengklasifikasikan gangguan mental berdasarkan teks yang diberikan. Dengan semakin banyaknya orang yang berbagi pemikiran dan perasaan secara online, ada potensi besar untuk menggunakan data teks ini untuk mendukung deteksi dini dan pemahaman yang lebih baik tentang kesehatan mental. Proyek ini mengeksplorasi beberapa pendekatan pembelajaran mesin, mulai dari model tradisional hingga *deep learning*, untuk mencapai tujuan klasifikasi ini.

## Sumber Data

Data yang digunakan dalam proyek ini adalah data yang berasal dari kaggle dapat diakses dari link berikut https://www.kaggle.com/datasets/cid007/mental-disorder-classification/data.
## Pra-pemrosesan Data

Semua notebook melakukan langkah-langkah pra-pemrosesan data yang serupa pada kolom teks, yang merupakan kunci untuk kinerja model NLP:

1.  **Pembersihan Teks**:
    * Menghapus URL, tag HTML, dan karakter non-alfabet.
    * Mengonversi teks ke huruf kecil.
    * Menghapus angka.
2.  **Tokenisasi**: Memecah teks menjadi kata-kata (token).
3.  **Penghapusan Stopwords**: Menghapus kata-kata umum yang tidak memberikan banyak makna (misalnya, "dan", "atau", "yang").
4.  **Stemming/Lemmatisasi**: Mengurangi kata-kata ke bentuk dasarnya (misalnya, "running" menjadi "run").
5.  **Vektorisasi Teks**: Mengonversi teks yang telah diproses menjadi representasi numerik yang dapat dipahami oleh model pembelajaran mesin. Metode yang umum digunakan meliputi:
    * **TF-IDF (Term Frequency-Inverse Document Frequency)**: Digunakan di sebagian besar model Scikit-learn dan XGBoost.
    * **Tokenisasi Keras/TensorFlow dan Padding Sekuens**: Digunakan untuk model *deep learning* (TensorFlow/Keras) untuk mengubah teks menjadi urutan angka dan memastikan panjang urutan yang seragam.

## Analisis Data Eksplorasi (EDA)

Meskipun tidak ada notebook EDA terpisah yang disediakan, langkah-langkah dalam notebook klasifikasi menunjukkan beberapa analisis dasar:

* **Pengecekan Nilai Hilang**: Memastikan tidak ada nilai yang hilang dalam kolom `text` atau `label`.
* **Distribusi Kelas**: Menampilkan jumlah sampel untuk setiap kategori gangguan mental (`label`) untuk memahami keseimbangan kelas. Ini penting untuk mengidentifikasi potensi masalah ketidakseimbangan data.

## Model Klasifikasi

Proyek ini mengeksplorasi berbagai model pembelajaran mesin untuk klasifikasi teks:

### Model Berbasis Scikit-learn (Logistic Regression, RandomForest, Naive Bayes)

Notebook `model_klasifikasi_mental_disorder.ipynb` dan `model_klasifikasi_mental_disorder copy.ipynb` mengimplementasikan dan mengevaluasi model-model ini:

* **Pembagian Data**: Data dibagi menjadi set pelatihan (training) dan pengujian (testing).
* **Vektorisasi**: `TfidfVectorizer` digunakan untuk mengubah teks menjadi fitur numerik.
* **Model yang Digunakan**:
    * **Logistic Regression**: Model linier yang baik untuk tugas klasifikasi biner dan multi-kelas.
    * **RandomForestClassifier**: Ensemble learning yang menggunakan banyak pohon keputusan.
    * **Multinomial Naive Bayes**: Algoritma klasifikasi probabilistik yang cocok untuk fitur diskrit (seperti hitungan kata).
* **Evaluasi**: Model dievaluasi menggunakan metrik seperti `accuracy_score`, `precision_score`, `recall_score`, dan `f1_score`, serta `classification_report` untuk laporan lengkap.

### Model XGBoost

Notebook `model_klasifikasi_mental_disorder_XGBoost.ipynb` berfokus pada model XGBoost:

* **Vektorisasi**: Juga menggunakan `TfidfVectorizer` untuk representasi teks.
* **Model**: `XGBClassifier` digunakan, yang merupakan implementasi algoritma *gradient boosting* yang sangat efisien dan kuat.
* **Evaluasi**: Metrik serupa (accuracy, precision, recall, f1-score) digunakan untuk menilai kinerja model.

### Model TensorFlow/Keras

Notebook `model_klasifikasi_tf.ipynb` mengimplementasikan model *deep learning* menggunakan TensorFlow dan Keras:

* **Tokenisasi Teks**: Menggunakan `Tokenizer` dari Keras untuk mengubah teks menjadi urutan integer.
* **Padding Sekuens**: `pad_sequences` digunakan untuk memastikan semua urutan memiliki panjang yang sama.
* **Arsitektur Model**: Model *deep learning* dasar dibangun, biasanya melibatkan:
    * `Embedding` layer: Untuk membuat representasi vektor padat dari kata-kata.
    * `GlobalAveragePooling1D` atau `Flatten` layer: Untuk merata-ratakan atau meratakan representasi kata.
    * `Dense` (lapisan terhubung penuh): Lapisan klasifikasi output.
* **Kompilasi dan Pelatihan**: Model dikompilasi dengan *optimizer* seperti Adam dan *loss function* yang sesuai (`SparseCategoricalCrossentropy`). Model dilatih selama beberapa *epoch*.
* **Evaluasi**: Dievaluasi menggunakan *accuracy* pada set pengujian.

### Model TensorFlow dengan 14 Fitur Teratas

Notebook `model_klasifikasi_tf_14fitur.ipynb` menyajikan variasi dari model TensorFlow/Keras, dengan fokus pada penggunaan subset fitur:

* **Pemilihan Fitur**: Asumsi adalah bahwa 14 fitur teratas (mungkin kata-kata atau *n-gram* penting) telah diidentifikasi dari analisis sebelumnya. Notebook ini melatih model TensorFlow hanya dengan fitur-fitur ini.
* **Proses Serupa**: Langkah-langkah tokenisasi, padding, arsitektur model, kompilasi, dan pelatihan pada dasarnya sama dengan `model_klasifikasi_tf.ipynb`, tetapi disesuaikan untuk input fitur yang lebih kecil.
* **Evaluasi**: Dievaluasi menggunakan *accuracy*.

## Cara Menggunakan (Pengaturan Lokal)

Untuk menjalankan proyek ini secara lokal, ikuti langkah-langkah berikut:

1.  **Kloning Repositori (jika berlaku)**:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Buat Lingkungan Virtual (Direkomendasikan)**:
    ```bash
    python -m venv .venv
    ```

3.  **Aktifkan Lingkungan Virtual**:
    * Di Windows:
        ```bash
        .venv\Scripts\activate
        ```
    * Di macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```

4.  **Instal Dependensi**:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn xgboost tensorflow keras nltk
    ```

5.  **Unduh Data NLTK (jika belum)**:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    ```

6.  **Jalankan Jupyter Notebooks**:
    * Buka setiap file `.ipynb` secara berurutan.
    * Jalankan setiap sel di dalam notebook untuk melihat langkah-langkah pra-pemrosesan, pelatihan model, dan evaluasi.
    * Anda mungkin perlu membuat file CSV data input (`data.csv` atau nama serupa) yang berisi kolom `text` dan `label` jika belum ada, berdasarkan format yang diharapkan oleh notebook.

## Inferensi 
hasil deploy model di hugging face bisa di coba pada link berikut https://formklasifikasi.netlify.app/ atau pada struktur folder ada inferensi dengan nama index.html, bisa diunduh dan run dilokal host

* note perhatikan lagi strukur folder nya
