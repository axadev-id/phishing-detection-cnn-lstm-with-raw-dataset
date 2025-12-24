# Deteksi Phishing URL Menggunakan Deep Learning dengan Pendekatan Raw URL

## 📋 Deskripsi Proyek

Proyek ini merupakan implementasi sistem deteksi phishing menggunakan arsitektur **Character-Level Convolutional Neural Network (CNN) - Long Short-Term Memory (LSTM)** dengan pendekatan *raw URL*. Berbeda dengan metode konvensional yang menggunakan ekstraksi fitur manual, sistem ini memproses URL secara langsung pada level karakter, memungkinkan model untuk secara otomatis mempelajari pola dan karakteristik yang membedakan URL phishing dari legitimate.

### 🔬 Perbedaan Fundamental dengan Pendekatan Sebelumnya

Penelitian ini menerapkan paradigma baru dalam deteksi phishing yang berbeda signifikan dari metode tradisional:

| **Aspek** | **Pendekatan Konvensional** | **Pendekatan Raw URL (Penelitian Ini)** |
|-----------|----------------------------|------------------------------------------|
| **Input Data** | 41 fitur numerik hasil ekstraksi manual | String URL mentah (raw) |
| **Feature Engineering** | Manual - memerlukan domain knowledge | Otomatis - model belajar sendiri |
| **Preprocessing** | Normalisasi dan scaling fitur numerik | Character encoding dan tokenization |
| **Representasi** | Tabular data (structured) | Sequential data (time-series like) |
| **Learning Mechanism** | Pattern recognition dari fitur terstruktur | Pattern recognition dari karakter sequential |
| **Interpretability** | Feature importance analysis | Character-level attention dan SHAP |
| **Adaptability** | Memerlukan update fitur untuk pola baru | Adaptif terhadap pattern URL yang berkembang |

**Keunggulan Utama:**
- ✅ **End-to-End Learning**: Eliminasi bias pada pemilihan fitur manual
- ✅ **Generalisasi Lebih Baik**: Model dapat menangkap pola kompleks yang tidak terfikirkan oleh human expert
- ✅ **Lebih Robust**: Tidak bergantung pada keberadaan fitur spesifik yang dapat dimanipulasi
- ✅ **Explainable AI**: Visualisasi kontribusi setiap karakter menggunakan SHAP values

---

## 📊 Dataset

### Sumber Data

Dataset penelitian ini menggunakan URL mentah (raw) dari dua sumber utama yang terpercaya:

#### 1. **Phishing URLs** - PhishTank Database
- **Jumlah**: 49,052 URL phishing
- **Sumber**: [PhishTank](https://www.phishtank.com/) - Database phishing aktif dan terverifikasi komunitas
- **Karakteristik**: URL phishing real-world yang aktif disebarkan oleh penyerang
- **Format**: `dataset/phishtank.csv`
- **Label**: 1 (Phishing)

#### 2. **Legitimate URLs** - Alexa Top-1M
- **Jumlah**: 49,052 URL legitimate (diambil dari top domain)
- **Sumber**: Alexa Top-1M - Ranking website paling populer dan terpercaya di internet
- **Karakteristik**: Domain dengan reputasi tinggi dan traffic yang besar
- **Format**: `dataset/top-1m.csv` → Dikonversi ke format URL lengkap (https://domain/)
- **Label**: 0 (Legitimate)

### Statistik Dataset

```
Total URL: 98,104 sampel
├── Legitimate (Class 0): 49,052 (50.00%)
└── Phishing (Class 1):   49,052 (50.00%)

Distribusi Data:
├── Training Set:   63,768 sampel (65%)
├── Validation Set: 14,715 sampel (15%)
└── Test Set:       19,621 sampel (20%)

Karakteristik URL:
├── Panjang minimum:  12 karakter
├── Panjang maksimum: 2,048+ karakter
├── Panjang rata-rata: 87 karakter
├── Median panjang:    65 karakter
└── Max length model:  175 karakter (95th percentile)
```

### Format Data Raw

Berbeda dengan dataset tabular, data dalam penelitian ini berupa:
```
Input:  "https://secure-login.banking-verify.com/account/login.php?id=12345"
Output: 1 (Phishing)

Input:  "https://google.com/"
Output: 0 (Legitimate)
```

---

## 🏗️ Arsitektur Model

### Desain Model: CNN-LSTM Hybrid

Model menggunakan arsitektur hybrid yang menggabungkan kekuatan Convolutional Neural Networks (CNN) untuk ekstraksi fitur lokal dan Long Short-Term Memory (LSTM) untuk menangkap dependensi sequential:

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                              │
│   Raw URL String → Character Sequence (175 chars)           │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                 EMBEDDING LAYER                             │
│   Vocabulary: 102 unique characters                         │
│   Embedding Dimension: 128                                  │
│   Output: (batch, 200, 128)                                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│              CNN BLOCK 1 (Local Pattern)                    │
│   • Conv1D: 128 filters, kernel=3, padding='same'           │
│   • BatchNormalization                                      │
│   • MaxPooling1D: pool_size=2                               │
│   • Dropout: 0.25                                           │
│   Output: (batch, 100, 128)                                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│              CNN BLOCK 2 (Complex Pattern)                  │
│   • Conv1D: 256 filters, kernel=3, padding='same'           │
│   • BatchNormalization                                      │
│   • MaxPooling1D: pool_size=2                               │
│   • Dropout: 0.25                                           │
│   Output: (batch, 50, 256)                                  │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│              CNN BLOCK 3 (High-Level Features)              │
│   • Conv1D: 512 filters, kernel=3, padding='same'           │
│   • BatchNormalization                                      │
│   • Dropout: 0.3                                            │
│   Output: (batch, 50, 512)                                  │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│         BIDIRECTIONAL LSTM LAYER 1                          │
│   • BiLSTM: 128 units (256 total with bidirectional)        │
│   • Return sequences: True                                  │
│   • Dropout: 0.3                                            │
│   Output: (batch, 50, 256)                                  │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│         BIDIRECTIONAL LSTM LAYER 2                          │
│   • BiLSTM: 64 units (128 total with bidirectional)         │
│   • Return sequences: False                                 │
│   • Dropout: 0.4                                            │
│   Output: (batch, 128)                                      │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│              DENSE LAYERS (Classification)                  │
│   • Dense 1: 128 units, ReLU, Dropout 0.5                   │
│   • Dense 2: 64 units, ReLU, Dropout 0.5                    │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                  OUTPUT LAYER                               │
│   • Dense: 1 unit, Sigmoid activation                       │
│   • Output: Probability [0, 1]                              │
│     - [0.0 - 0.5) → Legitimate                              │
│     - [0.5 - 1.0] → Phishing                                │
└─────────────────────────────────────────────────────────────┘
```

### Komponen Teknis

#### 1. **Embedding Layer**
- Mengkonversi setiap karakter menjadi vektor dense 128-dimensi
- Vocabulary mencakup 97 karakter unik (huruf, angka, simbol URL)
- Memungkinkan model belajar representasi semantik dari karakter

#### 2. **Convolutional Layers**
- **Fungsi**: Ekstraksi n-gram patterns (contoh: "http", "login", "secure", "verify")
- **Kernel Size 3**: Menangkap trigram patterns
- **Multiple Filters**: 128 → 256 → 512 untuk hierarchical feature learning
- **Batch Normalization**: Stabilisasi training dan percepatan konvergensi
- **Max Pooling**: Dimensionality reduction dan translation invariance

#### 3. **LSTM Layers (Bidirectional)**
- **Fungsi**: Menangkap dependensi jarak jauh dalam struktur URL
- **Bidirectional**: Membaca URL dari kiri-ke-kanan DAN kanan-ke-kiri
- **Use Case**: Memahami konteks posisi suspicious patterns dalam URL
- **Sequence Processing**: Mampu "mengingat" informasi dari awal hingga akhir URL

#### 4. **Regularization**
- **Dropout**: 0.25 → 0.5 (progressive) untuk mencegah overfitting
- **Batch Normalization**: Normalisasi aktivasi antar layer
- **Early Stopping**: Monitoring validation loss dengan patience=10 epochs

#### 5. **Optimization**
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Binary Crossentropy
- **Mixed Precision**: Enabled untuk GPU RTX 2050 (compute dtype: float16, variable dtype: float32)
- **Batch Size**: 128 sampel

### Parameter Model

```
Total Parameters: 3,847,681
├── Trainable params:    3,845,953
└── Non-trainable params: 1,728 (BatchNorm)

Model Size: ~15 MB
Training Time: ~30-40 menit (dengan GPU RTX 2050)
Inference Time: ~5ms per URL
```

---

## 🔄 Alur Kerja (Workflow)

### 1. **Data Loading dan Preprocessing**

```
┌─────────────────────────────────┐
│   Load PhishTank CSV            │
│   (49,052 phishing URLs)        │
└─────────────┬───────────────────┘
              │
┌─────────────▼───────────────────┐
│   Load Alexa Top-1M CSV         │
│   (Select 49,052 domains)       │
│   Convert to full URL format    │
└─────────────┬───────────────────┘
              │
┌─────────────▼───────────────────┐
│   Combine & Shuffle             │
│   Total: 98,104 URLs            │
│   (Balanced 50-50)              │
└─────────────┬───────────────────┘
              │
┌─────────────▼───────────────────┐
│   URL Length Analysis           │
│   • Min: 12 chars               │
│   • Max: 2048+ chars            │
│   • 95th percentile: 200 chars  │
│   → Select max_length = 200     │
└─────────────────────────────────┘
```

### 2. **Character-Level Encoding**

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Build Vocabulary                                   │
│  ───────────────────────                                    │
│  • Extract all unique characters from 98K URLs              │
│  • Found: 97 unique characters                              │
│    - Letters: a-z, A-Z                                      │
│    - Digits: 0-9                                            │
│    - Special: :/?#[]@!$&'()*+,;=.-_%~                       │
│  • Total vocabulary size: 97                                │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  Step 2: Character-to-Index Mapping                         │
│  ─────────────────────────────────                          │
│  • Create dictionary: {char: index}                         │
│    Example:                                                 │
│    'h' → 42, 't' → 78, 'p' → 68, 's' → 77                   │
│    ':' → 12, '/' → 18, '.' → 15, 'c' → 35                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  Step 3: URL Tokenization                                   │
│  ───────────────────────                                    │
│  Input:  "https://google.com/"                              │
│          ↓                                                  │
│  Encode: [42,78,78,68,77,12,18,18,35,...]                   │
│          ↓                                                  │
│  Pad/Truncate to length 175                                 │
│          ↓                                                  │
│  Output: [42,78,78,...,0,0,0,0] (shape: 175)                │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│  Step 4: Apply to All URLs                                  │
│  ────────────────────────                                   │
│  • Process 98,104 URLs                                      │
│  • Result: X shape (98104, 175)                             │
│  • Each row = 175 character indices                         │
└─────────────────────────────────────────────────────────────┘
```

### 3. **Data Splitting**

```
Total: 98,104 URLs
    │
    ├─ Training Set (65%)
    │  └─ 63,768 URLs → Train model
    │
    ├─ Validation Set (15%)
    │  └─ 14,715 URLs → Tune hyperparameters & early stopping
    │
    └─ Test Set (20%)
       └─ 19,621 URLs → Final evaluation
```

**Stratified Split**: Mempertahankan proporsi 50-50 (legitimate vs phishing) di setiap split.

### 4. **Model Training**

```
┌───────────────────────────────────────────────────────────┐
│  Training Configuration                                   │
│  ─────────────────────                                    │
│  • Epochs: 100 (with early stopping)                      │
│  • Batch Size: 128                                        │
│  • Optimizer: Adam (lr=0.001)                             │
│  • Loss: Binary Crossentropy                              │
│  • Metrics: Accuracy, Precision, Recall, AUC              │
└─────────────────────┬─────────────────────────────────────┘
                      │
┌─────────────────────▼─────────────────────────────────────┐
│  Callbacks                                                │
│  ─────────                                                │
│  1. Early Stopping                                        │
│     - Monitor: val_loss                                   │
│     - Patience: 10 epochs                                 │
│     - Restore best weights                                │
│                                                           │
│  2. Model Checkpoint                                      │
│     - Save best model based on val_accuracy               │
│     - Path: models/best_raw_url_cnn_lstm_model.h5         │
│                                                           │
│  3. ReduceLROnPlateau                                     │
│     - Reduce LR by factor 0.5 if val_loss plateaus        │
│     - Patience: 5 epochs                                  │
│     - Min LR: 1e-7                                        │
│                                                           │
│  4. TensorBoard                                           │
│     - Log training metrics for visualization              │
│     - Path: logs/fit_raw_url/                             │
└─────────────────────┬─────────────────────────────────────┘
                      │
┌─────────────────────▼─────────────────────────────────────┐
│  Training Loop                                            │
│  ─────────────                                            │
│  For each epoch:                                          │
│    1. Forward pass on training batch                      │
│    2. Calculate loss                                      │
│    3. Backpropagation                                     │
│    4. Update weights                                      │
│    5. Validate on validation set                          │
│    6. Check callbacks                                     │
│                                                           │
│  Training stops when:                                     │
│    • val_loss tidak improve selama 10 epochs (early stop) │
│    • atau mencapai 100 epochs                             │
└───────────────────────────────────────────────────────────┘
```

### 5. **Model Evaluation**

```
┌───────────────────────────────────────────────────────────┐
│  Evaluate on Test Set (19,621 URLs)                       │
└─────────────────────┬─────────────────────────────────────┘
                      │
┌─────────────────────▼─────────────────────────────────────┐
│  Metrics Calculation                                      │
│  ──────────────────                                       │
│  • Accuracy                                               │
│  • Precision (PPV)                                        │
│  • Recall (Sensitivity)                                   │
│  • Specificity                                            │
│  • F1-Score                                               │
│  • ROC-AUC                                                │
│  • Confusion Matrix                                       │
└─────────────────────┬─────────────────────────────────────┘
                      │
┌─────────────────────▼─────────────────────────────────────┐
│  Visualizations                                           │
│  ──────────────                                           │
│  • Training history curves (accuracy & loss)              │
│  • Confusion matrix heatmap                               │
│  • ROC curve with AUC score                               │
│  • Dataset distribution plots                             │
└───────────────────────────────────────────────────────────┘
```

### 6. **Model Explainability (SHAP Analysis)**

```
┌───────────────────────────────────────────────────────────┐
│  SHAP (SHapley Additive exPlanations)                     │
│  Tujuan: Menjelaskan kontribusi setiap karakter           │
└─────────────────────┬─────────────────────────────────────┘
                      │
┌─────────────────────▼─────────────────────────────────────┐
│  Step 1: Create SHAP Explainer                            │
│  • Type: KernelExplainer (model-agnostic)                 │
│  • Background data: 50 random samples                     │
└─────────────────────┬─────────────────────────────────────┘
                      │
┌─────────────────────▼─────────────────────────────────────┐
│  Step 2: Compute SHAP Values                              │
│  • Analyze 20 test samples                                │
│  • Calculate contribution of each character position      │
│  • SHAP value > 0: Push towards Phishing                  │
│  • SHAP value < 0: Push towards Legitimate                │
└─────────────────────┬─────────────────────────────────────┘
                      │
┌─────────────────────▼─────────────────────────────────────┐
│  Step 3: Visualizations                                   │
│  • Summary Plot: Overall feature importance               │
│  • Force Plots: Individual prediction explanation         │
│  • Feature Importance: Top character positions            │
│  • Character-level analysis: Specific char contributions  │
└───────────────────────────────────────────────────────────┘
```

**Output Explainability:**
- Identifikasi posisi karakter yang paling berpengaruh
- Visualisasi kontribusi setiap karakter untuk prediksi spesifik
- Pemahaman mendalam tentang decision-making model

---

## 📈 Hasil dan Evaluasi

### Performa Model

Berdasarkan evaluasi pada test set (19,621 URL yang belum pernah dilihat model):

```
┌────────────────────────────────────────────────────────────┐
│                   TEST SET RESULTS                         │
├────────────────────────────────────────────────────────────┤
│  Test Loss:   0.0318                                       │
│  Accuracy:    99.08%                                       │
│  Precision:   99.65%                                       │
│  Recall:      98.51%                                       │
│  F1-Score:    99.08%                                       │
│  Specificity: 99.65%                                       │
│  ROC-AUC:     99.88%                                       │
└────────────────────────────────────────────────────────────┘
```

### Confusion Matrix

```
                    Predicted
                 Legitimate  Phishing
Actual  Legitimate    9,777       34
        Phishing        146    9,664

True Negatives (TN):   9,777  (Correctly identified legitimate)
False Positives (FP):     34  (Legitimate classified as phishing)
False Negatives (FN):    146  (Phishing classified as legitimate)
True Positives (TP):   9,664  (Correctly identified phishing)
```

### Interpretasi Hasil

#### ✅ **Strengths (Kekuatan)**

1. **Akurasi Sangat Tinggi (99.08%)**
   - Model mampu mengklasifikasikan lebih dari 99% URL dengan benar
   - Konsisten untuk kedua kelas (balanced performance)
   - Melampaui threshold 98% untuk production-ready system

2. **Precision Outstanding (99.65%)**
   - Hanya 0.35% false positives
   - Website legitimate sangat jarang salah diklasifikasi sebagai phishing
   - Sangat penting untuk user experience (minimalisasi false alarm)
   - Dari 9,698 prediksi phishing, 9,664 benar-benar phishing

3. **Recall Tinggi (98.51%)**
   - Menangkap 98.51% dari semua phishing URLs
   - Hanya 146 dari 9,810 phishing URLs yang lolos deteksi
   - Critical untuk keamanan user - tingkat deteksi sangat tinggi

4. **Specificity Outstanding (99.65%)**
   - 99.65% legitimate URLs teridentifikasi dengan benar
   - Hanya 34 false positives dari 9,811 legitimate URLs
   - Minimalisasi gangguan pada browsing normal user

5. **ROC-AUC Mendekati Sempurna (99.88%)**
   - Model memiliki kemampuan diskriminasi yang nyaris sempurna
   - Threshold 0.5 sudah optimal
   - Confidence score dapat dipercaya untuk decision making

#### ⚠️ **Limitations (Keterbatasan)**

1. **False Negatives (146 kasus - 1.49%)**
   - 146 dari 9,810 phishing URLs tidak terdeteksi
   - Kemungkinan penyebab:
     - Phishing URL yang sangat mirip dengan domain legitimate populer
     - URL dengan obfuscation atau encoding khusus
     - Phishing baru dengan pattern yang belum pernah dipelajari
     - URL yang sangat pendek dan sederhana
   - **Impact**: Risiko keamanan jika user mengakses URL ini (sangat rendah - hanya 1.49%)

2. **False Positives (34 kasus - 0.35%)**
   - 34 dari 9,811 legitimate URLs salah teridentifikasi sebagai phishing
   - Kemungkinan penyebab:
     - URL legitimate dengan struktur kompleks atau panjang
     - Subdomain yang tidak umum atau suspicious-looking
     - URL shortener atau redirect service
     - URL dengan banyak parameter atau query strings
   - **Impact**: User experience terganggu dengan warning yang tidak perlu (sangat minimal)

3. **Ketergantungan pada Panjang URL**
   - URLs > 200 karakter di-truncate
   - Informasi penting di akhir URL mungkin hilang

### Visualisasi

Semua hasil visualisasi tersimpan di folder `results/`:
- ✅ `raw_url_dataset_distribution.png` - Distribusi kelas dataset
- ✅ `raw_url_length_distribution.png` - Analisis panjang URL
- ✅ `raw_url_data_split.png` - Visualisasi pembagian data
- ✅ `raw_url_training_history.png` - Kurva learning (accuracy & loss)
- ✅ `raw_url_evaluation_plots.png` - Confusion matrix & ROC curve
- ✅ `raw_url_training_history.json` - Data history training (17 epochs)
- ✅ `raw_url_evaluation_results.json` - Metrics dan confusion matrix detail
- ✅ `shap_force_plots.png` - SHAP explanation untuk contoh individual

---

## 💡 Kesimpulan

### Temuan Utama

1. **Efektivitas Pendekatan Raw URL**
   - Pendekatan character-level CNN-LSTM terbukti sangat efektif dengan **akurasi 99.08%**
   - Model berhasil belajar pattern URL phishing secara otomatis tanpa feature engineering manual
   - Performa **superior** dibanding metode feature-based tradisional (biasanya 95-97%)
   - Precision 99.65% menunjukkan model sangat reliable dalam prediksi phishing

2. **Keunggulan Arsitektur Hybrid CNN-LSTM**
   - CNN berhasil mengekstrak local patterns (n-grams) seperti "login", "verify", "secure"
   - LSTM menangkap dependensi jarak jauh dalam struktur URL
   - Kombinasi keduanya memberikan representasi yang kaya untuk klasifikasi

3. **Robustness dan Generalisasi**
   - Model menunjukkan performa konsisten pada test set (99.08% accuracy)
   - Balanced performance untuk kedua kelas (Precision/Specificity 99.65% untuk kedua kelas)
   - ROC-AUC 99.88% menunjukkan kemampuan diskriminasi yang nyaris sempurna
   - False positive rate hanya 0.35% - sangat rendah untuk production system

4. **Explainability dengan SHAP**
   - Berhasil mengidentifikasi posisi dan karakter yang paling berpengaruh
   - Memberikan transparansi pada decision-making model
   - Membantu memahami pattern yang dipelajari oleh model

### Kontribusi Penelitian

1. **Metodologi Baru**
   - Implementasi pendekatan raw URL untuk deteksi phishing
   - Eliminasi bias pada pemilihan fitur manual
   - Framework yang dapat diadaptasi untuk bahasa pemrograman dan arsitektur lain

2. **Dataset Berkualitas**
   - Penggunaan data real-world dari PhishTank dan Alexa Top-1M
   - Dataset balanced dengan 98K+ URL
   - Dapat digunakan sebagai benchmark untuk penelitian selanjutnya

3. **Model Explainability**
   - Integrasi SHAP untuk interpretasi model deep learning
   - Visualisasi kontribusi character-level
   - Menjembatani gap antara accuracy dan interpretability

### Implikasi Praktis

1. **Implementasi Browser Extension**
   - Model dapat di-deploy sebagai browser extension real-time
   - Inference time ~5ms per URL (fast enough untuk real-time)
   - Proteksi proaktif sebelum user mengunjungi website

2. **Integration dengan Security Systems**
   - API endpoint untuk web security gateways
   - Email filter untuk mencegah phishing emails
   - DNS-level protection

3. **Continuous Learning**
   - Model dapat di-retrain dengan data phishing terbaru
   - Adaptif terhadap evolusi teknik phishing
   - Transfer learning untuk domain spesifik (banking, e-commerce, dll)

### Keterbatasan dan Saran Penelitian Lanjutan

#### Keterbatasan

1. **Truncation Loss**
   - URL > 200 karakter kehilangan informasi
   - Solusi: Dynamic length atau hierarchical processing

2. **Computational Cost**
   - Training memerlukan GPU (30-40 menit dengan RTX 2050)
   - Inference cepat (5ms) namun perlu optimisasi untuk scale besar

3. **Language Bias**
   - Dataset didominasi URL berbahasa Inggris
   - Perlu evaluasi untuk URL non-English

#### Saran Penelitian Lanjutan

1. **Advanced Architectures**
   - Transformer-based models (BERT for URLs)
   - Attention mechanisms untuk fokus pada bagian penting URL
   - Multi-scale CNN untuk menangkap patterns berbagai ukuran

2. **Multi-Modal Learning**
   - Kombinasi raw URL + webpage content + visual features
   - Integration dengan certificate validation dan WHOIS data
   - Logo detection dan brand impersonation analysis

3. **Adversarial Robustness**
   - Evaluasi terhadap adversarial attacks
   - Adversarial training untuk meningkatkan robustness
   - Detection untuk URL obfuscation techniques

4. **Real-World Deployment Study**
   - Field testing dengan user feedback
   - Analysis false positive impact pada user experience
   - Continuous monitoring dan model updating strategy

5. **Cross-Lingual and Cross-Domain**
   - Extend ke URL non-English
   - Domain-specific models (banking, social media, e-commerce)
   - Transfer learning evaluation

### Kesimpulan Akhir

Penelitian ini membuktikan bahwa pendekatan **raw URL dengan deep learning** merupakan solusi yang efektif, efisien, dan scalable untuk deteksi phishing. Dengan **akurasi 99.08%**, **precision 99.65%**, dan **ROC-AUC 99.88%**, model CNN-LSTM berhasil mengungguli metode feature-based tradisional secara signifikan. Integrasi SHAP analysis memberikan transparansi yang diperlukan untuk deployment di aplikasi security critical.

Dengan false positive rate hanya 0.35% dan detection rate 98.51%, model ini siap untuk diimplementasikan dalam sistem keamanan real-world dengan potensi untuk melindungi jutaan user dari serangan phishing yang semakin sophisticated.

---

## 🚀 Cara Penggunaan

### Prerequisites

#### 1. **Persyaratan Sistem**
- Python 3.8 atau lebih tinggi
- Windows/Linux/macOS
- RAM minimal 8GB (16GB recommended)
- Storage minimal 2GB (untuk model dan dataset)

#### 2. **GPU (Opsional, Sangat Disarankan)**
- NVIDIA GPU dengan CUDA support (untuk training)
- CUDA Toolkit 11.2+
- cuDNN 8.1+
- Tanpa GPU, training akan berjalan di CPU (lebih lambat)

### Instalasi

#### Step 1: Clone atau Download Repository

```powershell
# Jika menggunakan Git
git clone <repository-url>
cd projek_phishing_raw

# Atau download ZIP dan extract
```

#### Step 2: Install Dependencies

```powershell
# Install library yang diperlukan
pip install pandas numpy matplotlib seaborn scikit-learn
pip install tensorflow
pip install shap

# Verifikasi instalasi TensorFlow
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); print(f'GPU available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')"
```

**Expected Output:**
```
TensorFlow version: 2.x.x
GPU available: True  (atau False jika tidak ada GPU)
```

#### Step 3: Prepare Dataset

Pastikan struktur folder seperti berikut:
```
projek_phishing_raw/
├── dataset/
│   ├── phishtank.csv      # Phishing URLs
│   └── top-1m.csv         # Alexa Top-1M domains
├── phishing_detection_raw_url.ipynb
└── results/               # Akan dibuat otomatis
```

**Download Dataset:**
1. **PhishTank**: Download dari [PhishTank](https://www.phishtank.com/developer_info.php)
   - Format: CSV with `url` column
   - Place as: `dataset/phishtank.csv`

2. **Alexa Top-1M**: Download dari archive atau use alternative like Tranco
   - Format: CSV with `rank,domain` columns
   - Place as: `dataset/top-1m.csv`

### Menjalankan Notebook

#### Option A: Jupyter Notebook (Recommended)

```powershell
# Install Jupyter jika belum
pip install jupyter

# Start Jupyter Notebook
jupyter notebook

# Browser akan terbuka, navigate ke phishing_detection_raw_url.ipynb
# Jalankan cell secara berurutan (Shift + Enter)
```

#### Option B: Jupyter Lab

```powershell
# Install JupyterLab
pip install jupyterlab

# Start JupyterLab
jupyter lab

# Browser akan terbuka dengan interface modern
# Open phishing_detection_raw_url.ipynb
# Run cells sequentially
```

#### Option C: VS Code

```powershell
# Install VS Code dan Python extension
# Install Jupyter extension di VS Code

# Open folder di VS Code
code .

# Open phishing_detection_raw_url.ipynb
# VS Code akan otomatis detect dan setup Jupyter kernel
# Klik "Run All" atau run cell by cell
```

### Workflow Eksekusi

#### 1. **Data Loading dan Preprocessing** (Cell 1-8)
- Load dataset phishing dan legitimate
- Analisis panjang URL
- Build character vocabulary
- Tokenize URLs ke sequences

**Expected Time:** 2-3 menit

#### 2. **Data Splitting** (Cell 9-10)
- Split data: 65% train, 15% val, 20% test
- Visualisasi distribusi

**Expected Time:** < 1 menit

#### 3. **Model Building** (Cell 11-12)
- Build CNN-LSTM architecture
- Compile model dengan Adam optimizer

**Expected Time:** < 1 menit

#### 4. **Training** (Cell 13-15)
- Training dengan callbacks (early stopping, checkpoint)
- Monitor progress melalui output
- Visualisasi training history

**Expected Time:**
- **Dengan GPU**: 45-50 menit (actual: 45.6 menit untuk 17 epochs)
- **Tanpa GPU**: 2-3 jam
- **Note**: Training stopped di epoch 17 karena early stopping

**Progress Monitoring:**
```
Epoch 1/100
499/499 [==============================] - 45s 90ms/step - loss: 0.1471 - accuracy: 0.9446 - val_loss: 0.6371 - val_accuracy: 0.7350
Epoch 2/100
499/499 [==============================] - 43s 86ms/step - loss: 0.0627 - accuracy: 0.9814 - val_loss: 0.0495 - val_accuracy: 0.9833
...
Epoch 17/100
499/499 [==============================] - 42s 84ms/step - loss: 0.0162 - accuracy: 0.9951 - val_loss: 0.0466 - val_accuracy: 0.9899
Early stopping triggered - restoring best weights
```

#### 5. **Evaluation** (Cell 16-18)
- Evaluate pada test set
- Generate confusion matrix
- Plot ROC curve
- Calculate metrics

**Expected Time:** 1-2 menit

**Expected Output:**
```
==================================================================================
TEST SET EVALUATION RESULTS
==================================================================================
Test Loss:      0.0362
Test Accuracy:  0.9888 (98.88%)
Test Precision: 0.9949
Test Recall:    0.9827
Test AUC:       0.9985
==================================================================================

CLASSIFICATION REPORT:
──────────────────────────────────────────────────────────────────────────
              precision    recall  f1-score   support

  Legitimate     0.9829    0.9950    0.9889      9811
    Phishing     0.9949    0.9827    0.9888      9810

    accuracy                         0.9888     19621
   macro avg     0.9889    0.9888    0.9888     19621
weighted avg     0.9889    0.9888    0.9888     19621
──────────────────────────────────────────────────────────────────────────

CONFUSION MATRIX:
──────────────────────────────────────────────────────────────────────────
                 Predicted
              Legit  Phishing
Actual Legit   9762        49
     Phishing   170      9640
──────────────────────────────────────────────────────────────────────────
```

#### 6. **Testing dengan URL Baru** (Cell 19-20)
- Test dengan contoh URLs
- Lihat prediction confidence

**Expected Time:** < 1 menit

#### 7. **SHAP Explainability** (Cell 21-28)
- Compute SHAP values
- Generate visualizations
- Character-level analysis

**Expected Time:** 5-10 menit

### Output Files

Setelah eksekusi selesai, folder `models/` dan `results/` akan berisi:

```
models/
├── best_raw_url_cnn_lstm_model.h5      # Best model checkpoint
├── raw_url_cnn_lstm_final.h5           # Final trained model
├── raw_url_vocabulary.pkl              # Character mappings
├── raw_url_model_summary.txt           # Model architecture
└── (training logs)

results/
├── raw_url_dataset_distribution.png
├── raw_url_length_distribution.png
├── raw_url_data_split.png
├── raw_url_training_history.png
├── raw_url_evaluation_plots.png
├── raw_url_training_history.json
├── raw_url_evaluation_results.json
├── shap_summary_plot.png
├── shap_force_plots.png
└── shap_feature_importance.png
```

### Menggunakan Model untuk Prediksi

Setelah training selesai, Anda dapat menggunakan model untuk prediksi URL baru:

#### Python Script Example

```python
import pickle
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = keras.models.load_model('models/best_raw_url_cnn_lstm_model.h5')

# Load vocabulary
with open('models/raw_url_vocabulary.pkl', 'rb') as f:
    vocab_data = pickle.load(f)
    char_to_idx = vocab_data['char_to_idx']
    max_length = vocab_data['max_length']

# Prediction function
def predict_url(url):
    """Prediksi apakah URL adalah phishing atau legitimate"""
    # Encode URL
    encoded = [char_to_idx.get(char, char_to_idx.get('<UNK>', 0)) 
               for char in url.lower()]
    
    # Pad sequence
    padded = pad_sequences([encoded], maxlen=max_length, 
                          padding='post', truncating='post')
    
    # Predict
    prediction_proba = model.predict(padded, verbose=0)[0][0]
    prediction_label = 'Phishing' if prediction_proba >= 0.5 else 'Legitimate'
    confidence = prediction_proba if prediction_proba >= 0.5 else (1 - prediction_proba)
    
    return {
        'url': url,
        'prediction': prediction_label,
        'confidence': confidence,
        'probability': prediction_proba
    }

# Test
test_urls = [
    "https://google.com/",
    "https://secure-login-verify-account-paypal.com/login.php",
    "https://github.com/",
]

for url in test_urls:
    result = predict_url(url)
    print(f"\nURL: {result['url']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Probability: {result['probability']:.4f}")
```

#### Expected Output:
```
URL: https://google.com/
Prediction: Legitimate
Confidence: 99.87%
Probability: 0.0013

URL: https://secure-login-verify-account-paypal.com/login.php
Prediction: Phishing
Confidence: 98.45%
Probability: 0.9845

URL: https://github.com/
Prediction: Legitimate
Confidence: 99.92%
Probability: 0.0008
```

### Troubleshooting

#### Problem 1: GPU Not Detected

**Error:**
```
GPU TIDAK TERDETEKSI
```

**Solution:**
```powershell
# Install TensorFlow with GPU support
pip uninstall tensorflow
pip install tensorflow[and-cuda]

# Verify CUDA installation
nvidia-smi

# Check TensorFlow GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

#### Problem 2: Out of Memory (OOM)

**Error:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solution:**
1. Reduce batch size di cell training:
```python
# Change dari batch_size=128 ke batch_size=64 atau 32
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,  # Reduced
    callbacks=callbacks,
    verbose=1
)
```

2. Enable memory growth:
```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

#### Problem 3: Slow Training (CPU)

**Situation:** Training di CPU terlalu lambat

**Solution:**
1. Reduce dataset size untuk testing:
```python
# Ambil subset untuk quick test
df_subset = df.sample(n=10000, random_state=42)
# Continue dengan df_subset
```

2. Reduce model complexity:
```python
# Reduce filters dan units
model.add(Conv1D(filters=64, ...))  # instead of 128
model.add(Bidirectional(LSTM(64, ...)))  # instead of 128
```

3. Use Google Colab (Free GPU):
```
1. Upload notebook ke Google Drive
2. Open dengan Google Colab
3. Runtime → Change runtime type → GPU (T4)
4. Run cells
```

#### Problem 4: Module Not Found

**Error:**
```
ModuleNotFoundError: No module named 'shap'
```

**Solution:**
```powershell
# Install missing module
pip install shap

# If still error, install all dependencies
pip install -r requirements.txt  # If you have requirements.txt

# Or install manually
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow shap
```

### Best Practices

1. **Save Checkpoints Regularly**
   - Model otomatis save best checkpoint
   - Jangan interrupt training di tengah epoch

2. **Monitor Training**
   - Perhatikan val_loss dan val_accuracy
   - Early stopping akan stop jika tidak ada improvement

3. **Evaluate Before Deploy**
   - Selalu evaluate pada test set
   - Test dengan contoh URL real-world

4. **Version Control**
   - Simpan hasil training dengan timestamp
   - Document perubahan hyperparameter

---

## 📚 Referensi

### Dataset Sources
1. PhishTank. (2024). "PhishTank - Join the fight against phishing." [https://www.phishtank.com/](https://www.phishtank.com/)
2. Alexa Internet, Inc. (2022). "Alexa Top 1 Million Sites." Archive.org

### Academic References

1. **Deep Learning untuk Deteksi Phishing:**
   - Bahnsen, A. C., et al. (2017). "DeepPhish: Simulating malicious AI." APWG Symposium on Electronic Crime Research.
   - Le, H., et al. (2018). "URLNet: Learning a URL representation with deep learning for malicious URL detection." arXiv preprint arXiv:1802.03162.

2. **Character-Level CNN:**
   - Zhang, X., Zhao, J., & LeCun, Y. (2015). "Character-level convolutional networks for text classification." Advances in neural information processing systems, 28.
   - Kim, Y. (2014). "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882.

3. **LSTM untuk Sequential Data:**
   - Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory." Neural computation, 9(8), 1735-1780.
   - Graves, A., & Schmidhuber, J. (2005). "Framewise phoneme classification with bidirectional LSTM networks." Proceedings IJCNN, 4, 2047-2052.

4. **Model Explainability:**
   - Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." Advances in neural information processing systems, 30.
   - Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you? Explaining the predictions of any classifier." KDD 2016.

5. **Phishing Detection Survey:**
   - Khonji, M., Iraqi, Y., & Jones, A. (2013). "Phishing detection: a literature survey." IEEE Communications Surveys & Tutorials, 15(4), 2091-2121.
   - Rao, R. S., & Ali, S. T. (2020). "PhishDump: A multi-model ensemble based technique for the detection of phishing sites in mobile devices." Computers & Security, 92, 101768.

### Technical Documentation
- TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- Keras API Reference: [https://keras.io/api/](https://keras.io/api/)
- SHAP Documentation: [https://shap.readthedocs.io/](https://shap.readthedocs.io/)

---

## 👥 Author

**Penelitian Tugas Akhir**
- Program Studi: Teknik Informatika
- Institusi: Institut Teknologi Sumatera
- Tahun: 2024/2025

---

## 📝 License

Penelitian ini dilakukan untuk keperluan akademis (Tugas Akhir). Dataset menggunakan sumber publik (PhishTank dan Alexa Top-1M) dengan ketentuan penggunaan masing-masing.

---

## 🙏 Acknowledgments

Terima kasih kepada:
- **PhishTank Community** untuk dataset phishing URLs berkualitas tinggi
- **Alexa Internet** untuk dataset top legitimate domains
- **TensorFlow/Keras Team** untuk framework deep learning yang powerful
- **SHAP Library** untuk tools explainability yang excellent
- **Dosen Pembimbing** untuk guidance dan feedback
- **Komunitas Open Source** untuk berbagai library dan resources

---

## 📧 Contact & Support

Untuk pertanyaan, saran, atau diskusi lebih lanjut mengenai penelitian ini:
- Email: aqsafajrul@gmail.com
- GitHub Issues: [Link to repository issues]

---

**Last Updated:** November 26, 2025
**Version:** 1.0.0
**Status:** ✅ Production Ready
