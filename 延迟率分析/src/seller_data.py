import pandas as pd
import numpy as np
from pathlib import Path

# 1. 设置路径并读取数据
BASE_DIR = Path(__file__).parent.parent
df = pd.read_csv(BASE_DIR / 'data' / "订单时间段分析清洗后.csv", encoding="utf-8-sig")

# 确保布尔列/标志列统一（如果你的数据已经是数值型 1/0，这句可以跳过）
# df['最终是否超时'] = df['最终是否超时'].astype(bool)
# df['发货是否超时'] = df['发货是否超时'].astype(bool)

# 2. 辅助列：计算“最终超时 且 发货也超时”的订单
df['最终超时_且_发货超时'] = ((df['最终是否超时'] == 1) | (df['最终是否超时'] == True)) & \
                             ((df['发货是否超时'] == 1) | (df['发货是否超时'] == True))

# 3. 按 seller_id 进行分组聚合
df_seller = df.groupby('seller_id').agg(
    超时订单数=('最终是否超时', 'sum'),
    总订单=('seller_id', 'size'),
    发货超时订单数=('发货是否超时', 'sum'),
    超时且发货超时数=('最终超时_且_发货超时', 'sum'), # 辅助列
    店铺平均评分=('订单评分', 'mean'),
    平均支付审核通过时间_天=('审核时间', 'mean'),
    平均承诺最晚发货时间_天=('预计最长发货周期', 'mean'),
    平均真实发货时间_天=('实际发货时长', 'mean'),
    平均承诺运送时间_天=('预计总运送周期', 'mean'),
    平均真实运送时间_天=('实际总运送周期', 'mean')
).reset_index()

# 4. 计算比率类指标（此时算出来的是 0-1 之间的小数）
df_seller['超时率'] = df_seller['超时订单数'] / df_seller['总订单']
df_seller['发货超时率'] = df_seller['发货超时订单数'] / df_seller['总订单']

df_seller['收货超时订单中发货超时占比'] = np.where(
    df_seller['超时订单数'] > 0, 
    df_seller['超时且发货超时数'] / df_seller['超时订单数'], 
    0 
)

# 5. 整理表头顺序并重命名
df_seller = df_seller[[
    'seller_id', '超时订单数', '总订单', '超时率', 
    '发货超时订单数', '发货超时率', '收货超时订单中发货超时占比', 
    '店铺平均评分', '平均支付审核通过时间_天', '平均承诺最晚发货时间_天', 
    '平均真实发货时间_天', '平均承诺运送时间_天', '平均真实运送时间_天'
]]

df_seller = df_seller.rename(columns={
    '平均支付审核通过时间_天': '平均支付审核通过时间/天',
    '平均承诺最晚发货时间_天': '平均承诺最晚发货时间/天',
    '平均真实发货时间_天': '平均真实发货时间/天',
    '平均承诺运送时间_天': '平均承诺运送时间/天',
    '平均真实运送时间_天': '平均真实运送时间/天'
})

# 6. 按数值大小进行多重降序排序 (必须在转字符串之前做)
df_seller = df_seller.sort_values(
    by=['超时订单数', '发货超时订单数', '收货超时订单中发货超时占比'],
    ascending=[False, False, False]
).reset_index(drop=True)

# 7. 格式化输出
# 处理百分比列：乘以100，保留三位小数，并且【不带 % 符号】
pct_cols = ['超时率', '发货超时率', '收货超时订单中发货超时占比']
for col in pct_cols:
    # f"{x * 100:.3f}" 会保留三位小数（即使末尾是0，如 12.300，也会保留）
    df_seller[col] = df_seller[col].fillna(0).map(lambda x: f"{x * 100:.3f}")

# 处理其他浮点数指标：保留三位小数
num_cols = [
    '店铺平均评分', '平均支付审核通过时间/天', '平均承诺最晚发货时间/天', 
    '平均真实发货时间/天', '平均承诺运送时间/天', '平均真实运送时间/天'
]
for col in num_cols:
    df_seller[col] = df_seller[col].map(lambda x: f"{x:.3f}" if pd.notnull(x) else x)

# 8. 导出文件
target_dir = BASE_DIR / 'processed_data'
target_dir.mkdir(parents=True, exist_ok=True)
file_path = target_dir / '延迟率与商家发货时间关系.csv'

# 仅导出一次，避免重复覆盖
df_seller.to_csv(file_path, index=False, encoding='utf-8-sig')

print(f"文件已成功导出至：{file_path}")
print("排序及格式化后的前 5 行数据：")
print(df_seller.head())