import pandas as pd
import numpy as np
from pathlib import Path

# 1. 设置路径并读取数据
BASE_DIR = Path(__file__).parent.parent
df = pd.read_csv(BASE_DIR / 'data' / "订单时间段分析清洗后.csv", encoding="utf-8-sig")

# ================= 核心修改步骤 =================
# 过滤掉没有实际最终延迟时间的行（比如未送达的订单）
df = df[df['实际最终延迟时间'].notna()].copy()

# 将实际最终延迟时间四舍五入为整数天，新列命名为 '超时天数' (int类型)
df['超时天数'] = df['实际最终延迟时间'].round(0).astype(int)
# =================================================

# 2. 按 超时天数 进行分组聚合
df_delay = df.groupby('超时天数').agg(
    总订单=('超时天数', 'size'),
    平均评分=('订单评分', 'mean'),
    平均审核时间=('审核时间', 'mean'),
    平均承诺发货时限=('预计最长发货周期', 'mean'),
    平均真实发货时间=('实际发货时长', 'mean'),
    平均承诺运送时间=('预计总运送周期', 'mean'),
    平均真实运送时间=('实际总运送周期', 'mean')
).reset_index()

# 3. 重命名列头，使其更符合分析报告
df_delay = df_delay.rename(columns={
    '总订单': '该天数总订单数',
    '平均评分': '该天数平均评分',
    '平均审核时间': '平均审核时间/天',
    '平均承诺发货时限': '平均承诺发货时限/天',
    '平均真实发货时间': '平均真实发货时间/天',
    '平均承诺运送时间': '平均承诺运送时间/天',
    '平均真实运送时间': '平均真实运送时间/天'
})

# 4. 排序：直接按 '超时天数' 升序排列（从负数提前天数，到正数延迟天数）
df_delay = df_delay.sort_values(by='超时天数', ascending=True).reset_index(drop=True)

# 5. 格式化输出
# 处理浮点数指标：保留三位小数
num_cols = [
    '该天数平均评分', '平均审核时间/天', '平均承诺发货时限/天', 
    '平均真实发货时间/天', '平均承诺运送时间/天', '平均真实运送时间/天'
]
for col in num_cols:
    df_delay[col] = df_delay[col].map(lambda x: f"{x:.3f}" if pd.notnull(x) else x)

# 6. 导出文件
target_dir = BASE_DIR / 'processed_data'
target_dir.mkdir(parents=True, exist_ok=True)
file_path = target_dir / '延迟天数与订单评分关系.csv'

df_delay.to_csv(file_path, index=False, encoding='utf-8-sig')

print(f"文件已成功导出至：{file_path}")
print("排序及格式化后的前 10 行数据：")
print(df_delay.head(10))