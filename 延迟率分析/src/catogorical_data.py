import pandas as pd
import numpy as np
from pathlib import Path

# 1. 设置路径并读取数据
BASE_DIR = Path(__file__).parent.parent
df = pd.read_csv(BASE_DIR / 'data' / "订单时间段分析清洗后.csv", encoding="utf-8-sig")

# ================= 新增：过滤商品类型为空的数据 =================
# 处理真实的 NaN 以及字符串形式的 'None'
df = df[df['商品类型'].notna()]
df = df[df['商品类型'].astype(str).str.lower() != 'none'].copy()
# ================================================================

# 2. 辅助列：计算“最终超时 且 发货也超时”的订单
df['最终超时_且_发货超时'] = ((df['最终是否超时'] == 1) | (df['最终是否超时'] == True)) & \
                             ((df['发货是否超时'] == 1) | (df['发货是否超时'] == True))

# 3. 按 商品类型 进行分组聚合 (临时命名，稍后统一重命名)
df_category = df.groupby('商品类型').agg(
    超时订单数=('最终是否超时', 'sum'),
    总订单=('商品类型', 'size'),
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
df_category['超时率'] = df_category['超时订单数'] / df_category['总订单']
df_category['发货超时率'] = df_category['发货超时订单数'] / df_category['总订单']

df_category['收货超时中发货超时占比'] = np.where(
    df_category['超时订单数'] > 0, 
    df_category['超时且发货超时数'] / df_category['超时订单数'], 
    0 
)

# 5. 整理表头顺序并重命名为更贴合“商品类型”业务场景的名字
df_category = df_category[[
    '商品类型', '超时订单数', '总订单', '超时率', 
    '发货超时订单数', '发货超时率', '收货超时中发货超时占比', 
    '平均评分', '平均审核时间', '平均承诺发货时限', 
    '平均真实发货时间', '平均承诺运送时间', '平均真实运送时间'
]]

df_category = df_category.rename(columns={
    '超时订单数': '该品类超时订单数',
    '总订单': '该品类总订单数',
    '超时率': '该品类超时率',
    '发货超时订单数': '该品类发货超时订单数',
    '发货超时率': '该品类发货超时率',
    '收货超时中发货超时占比': '该品类收货超时中发货超时占比',
    '平均评分': '该品类平均评分',
    '平均审核时间': '该品类平均审核时间/天',
    '平均承诺发货时限': '该品类平均承诺发货时限/天',
    '平均真实发货时间': '该品类平均真实发货时间/天',
    '平均承诺运送时间': '该品类平均承诺运送时间/天',
    '平均真实运送时间': '该品类平均真实运送时间/天'
})

# 6. 按数值大小进行多重降序排序
df_category = df_category.sort_values(
    by=['该品类超时订单数', '该品类发货超时订单数', '该品类收货超时中发货超时占比'],
    ascending=[False, False, False]
).reset_index(drop=True)

# 7. 格式化输出
# 处理百分比列：乘以100，保留三位小数，不带 %
pct_cols = ['该品类超时率', '该品类发货超时率', '该品类收货超时中发货超时占比']
for col in pct_cols:
    df_category[col] = df_category[col].fillna(0).map(lambda x: f"{x * 100:.3f}")

# 处理其他浮点数指标：保留三位小数
num_cols = [
    '该品类平均评分', '该品类平均审核时间/天', '该品类平均承诺发货时限/天', 
    '该品类平均真实发货时间/天', '该品类平均承诺运送时间/天', '该品类平均真实运送时间/天'
]
for col in num_cols:
    df_category[col] = df_category[col].map(lambda x: f"{x:.3f}" if pd.notnull(x) else x)

# 8. 导出文件
target_dir = BASE_DIR / 'processed_data'
target_dir.mkdir(parents=True, exist_ok=True)
file_path = target_dir / '延迟率与商品类型关系.csv'

df_category.to_csv(file_path, index=False, encoding='utf-8-sig')

print(f"文件已成功导出至：{file_path}")
print("排序及格式化后的前 5 行数据：")
print(df_category.head())