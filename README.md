# 🛡️ Hệ Thống Phát Hiện Tấn Công Mạng (IDS) - CICIDS2017

Dự án này sử dụng mô hình học máy **Random Forest** để phân loại các loại lưu lượng mạng (Network Traffic) dựa trên tập dữ liệu **CICIDS2017**. Hệ thống bao gồm từ bước khai phá dữ liệu, huấn luyện mô hình đến giao diện Dashboard dự đoán trực quan.

---

## 🚀 Hướng Dẫn Chạy Demo

### 1. Cài đặt môi trường
Trước tiên, hãy đảm bảo bạn đã cài đặt các thư viện cần thiết:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit plotly
```

### 2. Huấn luyện mô hình (Nếu chưa có file model)
Chạy script huấn luyện để xử lý dữ liệu và tạo ra file mô hình:
```bash
python3 step2_preprocess_train.py
```
*Script này sẽ tạo ra thư mục `models/` chứa file model và `visualizations/` chứa các biểu đồ phân tích.*

### 3. Chạy Dashboard Dự Đoán (Dùng để Demo)
Đây là phần quan trọng nhất để trình bày. Hãy chạy lệnh sau:
```bash
streamlit run app.py
```
Sau khi chạy, một trang web sẽ tự động mở ra tại: `http://localhost:8501`

---

## 📊 Kịch Bản Trình Bày (Cho Cô Giáo)

1. **Giới thiệu dữ liệu**: Mở thư mục `visualizations/` để cho cô xem:
   - `label_distribution.png`: Giải thích về độ mất cân bằng của dữ liệu.
   - `correlation_matrix.png`: Giải thích việc loại bỏ các đặc trưng thừa để tối ưu mô hình.
2. **Demo Trực Tiếp**:
   - Trên giao diện web, nhấn **"Browse files"**.
   - Chọn file `demo_sample.csv` (Đây là file mẫu 1400 dòng mình đã trích xuất sẵn để demo nhanh).
   - Giải thích các biểu đồ tròn và cột hiện ra: "Hệ thống đã nhận diện chính xác các loại tấn công như DoS, DDoS, Bots từ luồng dữ liệu thô."
3. **Kết luận**: Mô hình đạt độ chính xác **99.7%** và có khả năng triển khai thực tế.

---

## 📁 Cấu Trúc Thư Mục
- `app.py`: Mã nguồn giao diện Dashboard (Streamlit).
- `step2_preprocess_train.py`: Script xử lý dữ liệu và huấn luyện.
- `demo_sample.csv`: Dữ liệu mẫu dùng để demo dự đoán.
- `models/`: Lưu trữ model đã huấn luyện (`.pkl`).
- `visualizations/`: Chứa các biểu đồ phân tích dữ liệu (EDA).
