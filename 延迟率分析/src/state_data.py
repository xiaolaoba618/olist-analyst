import pandas as pd
import numpy as np
from pathlib import Path

# 1. 设置路径并读取数据
BASE_DIR = Path(__file__).parent.parent
df = pd.read_csv(BASE_DIR / 'data' / "订单时间段分析清洗后.csv", encoding="utf-8-sig")

# ================= 新增：过滤发货州为空的数据 =================
df = df[df['发货州'].notna()]
df = df[df['发货州'].astype(str).str.strip() != ''].copy()
# ==============================================================

# 2. 辅助列：计算“最终超时 且 发货也超时”的订单
df['最终超时_且_发货超时'] = ((df['最终是否超时'] == 1) | (df['最终是否超时'] == True)) & \
                             ((df['发货是否超时'] == 1) | (df['发货是否超时'] == True))

# 3. 按 发货州 进行分组聚合
df_state = df.groupby('发货州').agg(
    超时订单数=('最终是否超时', 'sum'),
    总订单=('发货州', 'size'),
    发货超时订单数=('发货是否超时', 'sum'),
    超时且发货超时数=('最终超时_且_发货超时', 'sum'), # 辅助列
    平均评分=('订单评分', 'mean'),
    平均审核时间=('审核时间', 'mean'),
    平均承诺发货时限=('预计最长发货周期', 'mean'),
    平均真实发货时间=('实际发货时长', 'mean'),
    平均承诺运送时间=('预计总运送周期', 'mean'),
    平均真实运送时间=('实际总运送周期', 'mean')
).reset_index()

# 4. 计算比率类指标（0-1 之间的小数）
df_state['超时率'] = df_state['超时订单数'] / df_state['总订单']
df_state['发货超时率'] = df_state['发货超时订单数'] / df_state['总订单']

df_state['收货超时中发货超时占比'] = np.where(
    df_state['超时订单数'] > 0, 
    df_state['超时且发货超时数'] / df_state['超时订单数'], 
    0 
)

# 5. 整理表头顺序并重命名为更贴合“发货州”业务场景的名字
df_state = df_state[[
    '发货州', '超时订单数', '总订单', '超时率', 
    '发货超时订单数', '发货超时率', '收货超时中发货超时占比', 
    '平均评分', '平均审核时间', '平均承诺发货时限', 
    '平均真实发货时间', '平均承诺运送时间', '平均真实运送时间'
]]

df_state = df_state.rename(columns={
    '超时订单数': '该州超时订单数',
    '总订单': '该州总订单数',
    '超时率': '该州超时率',
    '发货超时订单数': '该州发货超时订单数',
    '发货超时率': '该州发货超时率',
    '收货超时中发货超时占比': '该州收货超时中发货超时占比',
    '平均评分': '该州平均评分',
    '平均审核时间': '该州平均审核时间/天',
    '平均承诺发货时限': '该州平均承诺发货时限/天',
    '平均真实发货时间': '该州平均真实发货时间/天',
    '平均承诺运送时间': '该州平均承诺运送时间/天',
    '平均真实运送时间': '该州平均真实运送时间/天'
})

# 6. 按数值大小进行多重降序排序
df_state = df_state.sort_values(
    by=['该州超时订单数', '该州发货超时订单数', '该州收货超时中发货超时占比'],
    ascending=[False, False, False]
).reset_index(drop=True)

# 7. 格式化输出
# 处理百分比列：乘以100，保留三位小数，不带 %
pct_cols = ['该州超时率', '该州发货超时率', '该州收货超时中发货超时占比']
for col in pct_cols:
    df_state[col] = df_state[col].fillna(0).map(lambda x: f"{x * 100:.3f}")

# 处理其他浮点数指标：保留三位小数
num_cols = [
    '该州平均评分', '该州平均审核时间/天', '该州平均承诺发货时限/天', 
    '该州平均真实发货时间/天', '该州平均承诺运送时间/天', '该州平均真实运送时间/天'
]
for col in num_cols:
    df_state[col] = df_state[col].map(lambda x: f"{x:.3f}" if pd.notnull(x) else x)

# 8. 导出文件
target_dir = BASE_DIR / 'processed_data'
target_dir.mkdir(parents=True, exist_ok=True)
file_path = target_dir / '延迟率与发货州关系.csv'

df_state.to_csv(file_path, index=False, encoding='utf-8-sig')

print(f"文件已成功导出至：{file_path}")
print("排序及格式化后的前 5 行数据：")
print(df_state.head())
