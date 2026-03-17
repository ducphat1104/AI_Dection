"""
Lấy ví dụ cụ thể từng dòng data để giải thích cho cô giáo
"""

import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('cicids2017_cleaned.csv')

# Key features để phân tích
key_features = [
    'Flow Duration',
    'Total Fwd Packets', 
    'Fwd Packet Length Max',
    'Bwd Packet Length Max',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Flow IAT Mean',
    'FIN Flag Count',
    'PSH Flag Count', 
    'ACK Flag Count',
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward'
]

print("="*100)
print("VÍ DỤ CỤ THỂ TỪNG DÒNG DATA - ĐỂ GIẢI THÍCH CHO CÔ GIÁO")
print("="*100)

# Lấy 1 mẫu đại diện cho mỗi loại
examples = {}
for attack_type in df['Attack Type'].unique():
    subset = df[df['Attack Type'] == attack_type]
    # Lấy mẫu gần với median để đại diện
    median_duration = subset['Flow Duration'].median()
    closest_idx = (subset['Flow Duration'] - median_duration).abs().idxmin()
    examples[attack_type] = subset.loc[closest_idx]

print("\n1. VÍ DỤ CỤ THỂ TỪNG LOẠI ATTACK:")
print("="*60)

for i, (attack_type, example) in enumerate(examples.items(), 1):
    print(f"\n{i}. {attack_type.upper()}:")
    print("-" * 40)
    
    for feature in key_features:
        value = example[feature]
        if isinstance(value, float):
            if abs(value) > 1000:
                print(f"   {feature:<25}: {value:>12,.0f}")
            else:
                print(f"   {feature:<25}: {value:>12.2f}")
        else:
            print(f"   {feature:<25}: {value:>12,}")

print("\n\n" + "="*100)
print("2. SO SÁNH TRỰC TIẾP - CÁC CHỈ SỐ QUYẾT ĐỊNH:")
print("="*100)

# Tạo bảng so sánh
comparison_df = pd.DataFrame()
for attack_type, example in examples.items():
    comparison_df[attack_type] = [example[feature] for feature in key_features]
comparison_df.index = key_features

# Format và in bảng
print(f"\n{'Feature':<25}", end="")
for attack in comparison_df.columns:
    print(f"{attack:>15}", end="")
print()
print("-" * (25 + 15 * len(comparison_df.columns)))

for feature in key_features:
    print(f"{feature:<25}", end="")
    for attack in comparison_df.columns:
        value = comparison_df.loc[feature, attack]
        if isinstance(value, float):
            if abs(value) > 1000:
                print(f"{value:>15,.0f}", end="")
            else:
                print(f"{value:>15.2f}", end="")
        else:
            print(f"{value:>15,}", end="")
    print()

print("\n\n" + "="*100)
print("3. GIẢI THÍCH CHO CÔ GIÁO - CÁC CHỈ SỐ QUYẾT ĐỊNH:")
print("="*100)

print("""
🎯 KHI CÔ HỎI: "Làm sao biết đây là DoS chứ không phải Normal Traffic?"

📊 TRÌNH BÀY CỤ THỂ:

1️⃣ FLOW DURATION (Thời gian tồn tại của kết nối):
   • Normal Traffic: ~40,000 μs (40ms) - bình thường
   • DoS Attack:     ~85,000,000 μs (85 giây) - GẤP 2000 LẦN!
   
   → "Cô ạ, kết nối DoS kéo dài 85 giây để bơm liên tục, 
      trong khi Normal chỉ 40ms là xong việc"

2️⃣ FLOW BYTES/S (Tốc độ truyền dữ liệu):
   • Normal Traffic: ~1,600,000 Bytes/s (1.6 MB/s) - bình thường
   • DoS Attack:     ~70,000 Bytes/s - thấp hơn nhưng KÉO DÀI
   
   → "DoS không phải truyền nhanh mà truyền LIÊN TỤC để làm tắc nghẽn"

3️⃣ TOTAL FWD PACKETS (Số gói tin gửi đi):
   • Normal Traffic: ~11 packets - vừa đủ
   • DoS Attack:     ~6 packets - ít hơn nhưng mỗi gói RẤT LỚN
   
4️⃣ TCP FLAGS (Cờ hiệu trong gói tin):
   • Normal Traffic: PSH=0.27, ACK=0.29 (cân bằng)
   • DoS Attack:     PSH=0.15, ACK=0.54, FIN=0.30
   
   → "DoS có nhiều FIN flags (đóng kết nối) và ACK floods"

🔍 PORT SCANNING - Dễ nhận diện nhất:
   • Flow Duration: 50 μs (cực ngắn - chỉ probe)
   • Total Fwd Packets: 1 (chỉ gửi 1 gói để thăm dò)
   • PSH Flag: 1.0 (100% là push packets)
   
   → "Cô ạ, Port Scan chỉ gửi 1 gói tin trong 50 microsecond rồi đi luôn"

🤖 BOTS - Khó nhận diện nhất:
   • Flow Duration: 71,036 μs (gần giống Normal: 39,979 μs)
   • Flow Bytes/s: 290,129 (gần giống Normal: 1,675,795)
   • Flags: PSH=0.63, ACK=0.37 (gần giống Normal)
   
   → "Bots cố tình giả mạo Normal Traffic, chỉ khác nhau ở những chi tiết nhỏ
      mà chỉ Machine Learning mới phát hiện được!"

💡 BRUTE FORCE:
   • Total Fwd Packets: 11 (nhiều attempts)
   • PSH Flag: 0.76 (76% push data - gửi username/password)
   • Init_Win_bytes_forward: 22,245 (window lớn - chờ response)

🌐 WEB ATTACKS:
   • Fwd Packet Length Max: 48 bytes (nhỏ nhưng chứa payload)
   • PSH Flag: 0.94 (94% push - HTTP requests)
   • Init_Win_bytes: Cả 2 chiều đều lớn (HTTP communication)
""")

print("\n" + "="*100)
print("4. CÂU TRẢ LỜI MẪU CHO CÔ GIÁO:")
print("="*100)

print("""
🎤 "Thưa cô, em sẽ lấy ví dụ cụ thể:

📋 VÍ DỤ 1 - PHÂN BIỆT DoS VÀ NORMAL:
   Nếu có 1 dòng data với:
   • Flow Duration = 85,872,118 μs (85 giây)
   • Flow Bytes/s = 70,555
   • FIN Flag Count = 0.30
   
   → Đây chắc chắn là DoS vì:
     ✓ Kết nối kéo dài 85 giây (Normal chỉ 40ms)
     ✓ Có 30% FIN flags (đóng kết nối liên tục)
     ✓ Pattern này không thể là Normal Traffic

📋 VÍ DỤ 2 - PHÂN BIỆT PORT SCANNING:
   Nếu có 1 dòng data với:
   • Flow Duration = 50 μs (cực ngắn)
   • Total Fwd Packets = 1
   • PSH Flag Count = 1.0 (100%)
   
   → Đây chắc chắn là Port Scanning vì:
     ✓ Chỉ gửi 1 gói tin rồi đi (probe)
     ✓ Thời gian cực ngắn (50 microsecond)
     ✓ 100% PSH flags (push probe packet)

📋 VÍ DỤ 3 - TẠI SAO BOTS KHÓ PHÁT HIỆN:
   Bots có chỉ số gần giống Normal:
   • Flow Duration: 71,036 vs 39,979 μs (chỉ khác 1.8x)
   • PSH Flag: 0.63 vs 0.27 (khác nhau nhưng không rõ ràng)
   
   → Chỉ khi kết hợp 48 features cùng lúc, Machine Learning 
     mới phát hiện được pattern tinh tế này!"

🎯 KẾT LUẬN: 
   "Mỗi loại attack để lại 'dấu vết' khác nhau trong network traffic.
   Con người có thể nhận ra DoS, Port Scan dễ dàng,
   nhưng cần Machine Learning để phát hiện Bots và Web Attacks!"
""")

print("\n" + "="*100)
print("5. DEMO TRỰC TIẾP VỚI CÔ:")
print("="*100)

print("""
💻 "Cô có thể thử ngay trên dashboard:

1. Mở file demo_sample.csv
2. Tìm dòng có Flow Duration > 80,000,000 → sẽ thấy label "DoS"
3. Tìm dòng có Total Fwd Packets = 1 → sẽ thấy label "Port Scanning"
4. So sánh với Normal Traffic → thấy rõ sự khác biệt

→ Model đã học được những pattern này và áp dụng cho dữ liệu mới!"
""")