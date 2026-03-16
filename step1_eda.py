"""
Step 1: Exploratory Data Analysis (EDA)
Dataset: CICIDS2017 (cleaned)
Mục tiêu:
  - Hiểu phân phối dữ liệu
  - Phát hiện inf/NaN
  - Tìm features tương quan cao → quyết định drop
  - Visualize phân phối label và một số feature quan trọng
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ─── PATHS ─────────────────────────────────────────────────────────────────────
DATASET_PATH = 'cicids2017_cleaned.csv'
OUT_DIR      = 'visualizations'
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 1. LOAD & OVERVIEW ────────────────────────────────────────────────────────
print("=" * 60)
print("1. TỔNG QUAN DỮ LIỆU")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)
print(f"   Shape       : {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"\n   Dtypes:\n{df.dtypes.value_counts().to_string()}")
print(f"\n   Columns: {list(df.columns)}")

# ─── 2. KIỂM TRA CHẤT LƯỢNG DỮ LIỆU ──────────────────────────────────────────
print("\n" + "=" * 60)
print("2. KIỂM TRA CHẤT LƯỢNG DỮ LIỆU")
print("=" * 60)

# 2a. Null values
null_counts = df.isnull().sum()
print(f"\n   Tổng null values : {null_counts.sum():,}")
if null_counts.sum() > 0:
    print(null_counts[null_counts > 0])
else:
    print("   → Không có null values.")

# 2b. Inf values (phổ biến trong CICIDS2017 do chia cho flow duration = 0)
num_cols = df.select_dtypes(include=[np.number]).columns
inf_counts = np.isinf(df[num_cols]).sum()
total_inf  = inf_counts.sum()
print(f"\n   Tổng inf values  : {total_inf:,}")
if total_inf > 0:
    print("   Các cột có inf:")
    print(inf_counts[inf_counts > 0].to_string())
    print("\n   → Nguyên nhân: Flow Bytes/s = Total Bytes / Flow Duration")
    print("     Khi Flow Duration = 0 (gói tin đến/đi gần như cùng lúc) → chia cho 0 → inf")
    print("     Xử lý: replace inf → NaN → fillna(median)")
else:
    print("   → Không có inf values (đã được xử lý trong bước clean trước).")

# 2c. Duplicate rows
dup_count = df.duplicated().sum()
print(f"\n   Duplicate rows   : {dup_count:,}")

# ─── 3. PHÂN PHỐI LABEL ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. PHÂN PHỐI NHÃN (ATTACK TYPE)")
print("=" * 60)

label_counts = df['Attack Type'].value_counts()
label_pct    = df['Attack Type'].value_counts(normalize=True) * 100

print(f"\n{'Label':<20} {'Count':>10} {'Percent':>10}")
print("-" * 42)
for label in label_counts.index:
    print(f"   {label:<18} {label_counts[label]:>10,} {label_pct[label]:>9.2f}%")

print(f"\n   → Dữ liệu MẤT CÂN BẰNG NẶNG:")
print(f"     Normal Traffic chiếm {label_pct['Normal Traffic']:.1f}% tổng số mẫu.")
print(f"     Bots chỉ có {label_counts['Bots']:,} mẫu ({label_pct['Bots']:.2f}%) — class nhỏ nhất.")
print(f"   → Giải pháp: dùng class_weight='balanced' trong Random Forest")
print(f"     và đánh giá bằng F1-macro thay vì accuracy đơn thuần.")

# Plot label distribution
plt.figure(figsize=(10, 5))
colors = ['#2ecc71' if l == 'Normal Traffic' else '#e74c3c' for l in label_counts.index]
bars = plt.bar(label_counts.index, label_counts.values, color=colors, edgecolor='white')
plt.title('Phân Phối Nhãn trong CICIDS2017', fontsize=14, fontweight='bold')
plt.xlabel('Attack Type')
plt.ylabel('Số lượng mẫu')
plt.xticks(rotation=20, ha='right')
for bar, val in zip(bars, label_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
             f'{val:,}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'label_distribution.png'), dpi=150)
plt.close()
print(f"\n   Saved → visualizations/label_distribution.png")

# ─── 4. THỐNG KÊ MÔ TẢ ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. THỐNG KÊ MÔ TẢ (NUMERIC FEATURES)")
print("=" * 60)

desc = df[num_cols].describe().T
desc['cv'] = desc['std'] / (desc['mean'].abs() + 1e-9)  # coefficient of variation
print(desc[['mean', 'std', 'min', 'max', 'cv']].round(2).to_string())

# ─── 5. CORRELATION ANALYSIS → QUYẾT ĐỊNH DROP ────────────────────────────────
print("\n" + "=" * 60)
print("5. PHÂN TÍCH TƯƠNG QUAN → QUYẾT ĐỊNH DROP FEATURES")
print("=" * 60)

corr_matrix = df[num_cols].corr().abs()

# Tìm các cặp có tương quan cao (> 0.85), loại bỏ self-correlation
upper_tri = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
high_corr_pairs = (
    upper_tri.stack()
    .reset_index()
    .rename(columns={'level_0': 'Feature A', 'level_1': 'Feature B', 0: 'Correlation'})
    .query('Correlation > 0.85')
    .sort_values('Correlation', ascending=False)
)

print(f"\n   Các cặp features có tương quan > 0.85:")
print(f"\n{'Feature A':<30} {'Feature B':<30} {'Corr':>6}")
print("-" * 68)
for _, row in high_corr_pairs.iterrows():
    print(f"   {row['Feature A']:<28} {row['Feature B']:<28} {row['Correlation']:>6.3f}")

print("""
   → Quyết định DROP chỉ 4 features có corr > 0.90 (ngưỡng "chắc chắn dư thừa"):
     DROP 'Fwd Packet Length Mean'  → giữ 'Fwd Packet Length Max'  (corr=0.89)
     DROP 'Bwd Packet Length Mean'  → giữ 'Bwd Packet Length Max'  (corr=0.96)
     DROP 'Average Packet Size'     → giữ 'Packet Length Mean'     (corr=1.00, hoàn toàn trùng)
     DROP 'Fwd IAT Mean'            → giữ 'Flow IAT Mean'          (corr=0.90)

   → Tại sao KHÔNG drop thêm dù còn nhiều cặp corr > 0.85?
     1. Random Forest kháng đa cộng tuyến: RF chọn ngẫu nhiên subset features
        khi chia nhánh → nếu 2 cột giống nhau, nó tự chọn 1. Corr vừa phải
        (0.7–0.85) không ảnh hưởng đáng kể đến accuracy.
     2. Signal preservation: Trong an ninh mạng, sự khác biệt giữa Normal và
        Attack đôi khi nằm ở sai số rất nhỏ. Drop quá nhiều có thể làm mất
        tín hiệu phân biệt các class hiếm như Bots (0.08%) hay Web Attacks (0.09%).
     3. Giữ Max thay vì Mean: Các giá trị cực đại (peak) phản ánh hành vi tấn
        công rõ hơn (ví dụ bơm gói tin cực lớn trong DoS), nên ưu tiên giữ Max.
""")

# Plot correlation heatmap (chỉ lấy subset features để dễ đọc)
selected_features = [
    'Fwd Packet Length Max', 'Fwd Packet Length Mean',
    'Bwd Packet Length Max', 'Bwd Packet Length Mean',
    'Packet Length Mean', 'Average Packet Size',
    'Flow IAT Mean', 'Fwd IAT Mean',
    'Flow Bytes/s', 'Flow Packets/s',
    'Flow Duration', 'Total Fwd Packets',
]
corr_sub = df[selected_features].corr()

plt.figure(figsize=(12, 9))
mask = np.triu(np.ones_like(corr_sub, dtype=bool))
sns.heatmap(corr_sub, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, annot_kws={'size': 8})
plt.title('Correlation Matrix – Selected Features\n(Cơ sở để quyết định drop features thừa)',
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'correlation_matrix.png'), dpi=150)
plt.close()
print(f"   Saved → visualizations/correlation_matrix.png")

# ─── 6. PHÂN PHỐI FLOW DURATION THEO ATTACK TYPE ──────────────────────────────
print("\n" + "=" * 60)
print("6. PHÂN PHỐI FLOW DURATION THEO ATTACK TYPE")
print("=" * 60)

# Log scale vì Flow Duration có outlier rất lớn
df_plot = df.copy()
df_plot['Flow Duration (log)'] = np.log1p(df_plot['Flow Duration'])

plt.figure(figsize=(12, 6))
order = df['Attack Type'].value_counts().index
sns.boxplot(data=df_plot, x='Attack Type', y='Flow Duration (log)',
            order=order, palette='Set2')
plt.title('Phân Phối Flow Duration theo Attack Type (log scale)', fontsize=13, fontweight='bold')
plt.xlabel('Attack Type')
plt.ylabel('log(1 + Flow Duration)')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'flow_duration_boxplot.png'), dpi=150)
plt.close()
print(f"   DoS/DDoS thường có flow duration ngắn (tấn công nhanh, nhiều gói).")
print(f"   Normal Traffic có phân phối rộng hơn.")
print(f"   Saved → visualizations/flow_duration_boxplot.png")

# ─── 7. TÓM TẮT KẾT QUẢ EDA ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. TÓM TẮT KẾT QUẢ EDA")
print("=" * 60)
print(f"""
   Dataset : {df.shape[0]:,} mẫu, {df.shape[1]} cột (52 features + 1 label)
   Classes : 7 loại (Normal + 6 loại tấn công)
   
   Vấn đề phát hiện:
   ├── Mất cân bằng nặng: Normal chiếm 83.1%, Bots chỉ 0.08%
   └── 4 features thừa (tương quan > 0.90 với feature khác)
   
   Hành động cho Step 2:
   ├── Drop 4 redundant features
   ├── Dùng stratify=y khi split để giữ tỉ lệ class
   └── Dùng class_weight='balanced' trong Random Forest
""")

print("✅ EDA hoàn tất!")
