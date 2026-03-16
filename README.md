# 🛡️ Hệ Thống Phát Hiện Tấn Công Mạng (IDS) - CICIDS2017

Dự án này sử dụng mô hình học máy **Random Forest + SMOTE** để phân loại các loại lưu lượng mạng (Network Traffic) dựa trên tập dữ liệu **CICIDS2017**. Hệ thống bao gồm từ bước khai phá dữ liệu, huấn luyện mô hình đến giao diện Dashboard dự đoán trực quan.

## 📈 Kết Quả Mô Hình

- **Accuracy**: 99.83%
- **F1-score (macro)**: 0.95
- **F1-score (weighted)**: 0.998

### Chi tiết theo từng loại tấn công:
| Attack Type | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Bots | 0.51 | 0.94 | 0.66 | 292 |
| Brute Force | 1.00 | 1.00 | 1.00 | 1,373 |
| DDoS | 1.00 | 1.00 | 1.00 | 19,202 |
| DoS | 1.00 | 1.00 | 1.00 | 29,062 |
| Normal Traffic | 1.00 | 1.00 | 1.00 | 314,259 |
| Port Scanning | 0.99 | 1.00 | 0.99 | 13,604 |
| Web Attacks | 0.98 | 0.98 | 0.98 | 321 |

---

## 🚀 Hướng Dẫn Chạy Demo

### 1. Cài đặt môi trường
Trước tiên, hãy đảm bảo bạn đã cài đặt các thư viện cần thiết:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit plotly imbalanced-learn
```

### 2. Chạy EDA (Khám phá dữ liệu - Optional)
Để hiểu rõ về dataset và quyết định preprocessing:
```bash
python3 step1_eda.py
```
*Script này tạo ra các biểu đồ phân tích trong thư mục `visualizations/`*

### 3. Huấn luyện mô hình
Chạy script huấn luyện để xử lý dữ liệu và tạo ra file mô hình:
```bash
python3 step2_preprocess_train.py
```
*Script này sẽ:*
- Drop 4 features thừa (correlation > 0.90)
- Áp dụng SMOTE để cân bằng class Bots và Web Attacks
- Train Random Forest (150 trees, max_depth=25)
- Lưu model vào `models/` và visualizations vào `visualizations/`

### 4. Đánh giá trên Test Set
```bash
python3 step3_final_test.py
```

### 5. Chạy Dashboard Dự Đoán (Dùng để Demo)
Đây là phần quan trọng nhất để trình bày. Hãy chạy lệnh sau:
```bash
streamlit run app.py
```
Sau khi chạy, một trang web sẽ tự động mở ra tại: `http://localhost:8501`

---

## � Cấu Trảúc Thư Mục
- `step1_eda.py`: Script khám phá dữ liệu (EDA) - phân tích correlation, phân phối label, v.v.
- `step2_preprocess_train.py`: Script xử lý dữ liệu, áp dụng SMOTE, và huấn luyện Random Forest.
- `step3_final_test.py`: Đánh giá model trên test set.
- `app.py`: Mã nguồn giao diện Dashboard (Streamlit).
- `predict_utility.py`: Utility function để predict từ file CSV mới.
- `demo_sample.csv`: Dữ liệu mẫu dùng để demo dự đoán.
- `models/`: Lưu trữ model đã huấn luyện (`.pkl`).
- `visualizations/`: Chứa các biểu đồ phân tích dữ liệu (EDA).

## 🔬 Kỹ Thuật Sử Dụng

### Preprocessing
- Drop 4 redundant features (correlation > 0.90)
- StandardScaler cho numerical features
- Stratified split 70/15/15 (train/val/test)

### Handling Imbalance
- SMOTE: Oversample Bots (1,364 → 4,000) và Web Attacks (1,500 → 4,500)
- class_weight='balanced' trong Random Forest

### Model
- Random Forest Classifier
  - n_estimators: 150
  - max_depth: 25
  - min_samples_split: 5
  - class_weight: 'balanced'

### Evaluation Metrics
- Accuracy: Tổng quan performance
- F1-macro: Quan trọng nhất với imbalanced data (trung bình không trọng số)
- F1-weighted: Trung bình có trọng số theo số lượng samples
- Confusion Matrix: Phân tích chi tiết từng class
