import pandas as pd
import numpy as np
from pathlib import Path

# 1. 设置路径并读取数据
BASE_DIR = Path(__file__).parent.parent
# 文件名保持不变
df = pd.read_csv(BASE_DIR / 'data' / "订单时间段分析清洗后.csv", encoding="utf-8-sig")

# 确保必备的布尔/数值列存在（如果原表中没有重新生成）
# 假设前面的代码已经生成了“超时因素”列。如果没有，你需要先运行前面生成“超时因素”的逻辑。

# 2. 基础维度聚合（保留原有的评分和时间均值）
df_seller_base = df.groupby('seller_id').agg(
    总订单=('seller_id', 'size'),
    店铺平均评分=('订单评分', 'mean'),
    平均支付审核通过时间_天=('审核时间', 'mean'),
    平均承诺最晚发货时间_天=('预计最长发货周期', 'mean'),
    平均真实发货时间_天=('实际发货时长', 'mean'),
    平均承诺运送时间_天=('预计总运送周期', 'mean'),
    平均真实运送时间_天=('实际总运送周期', 'mean')
).reset_index()

# 3. 核心新增：按订单级“超时因素”进行透视统计
# 使用 crosstab 生成每个商家各类超时原因的订单数
df_factors = pd.crosstab(df['seller_id'], df['超时因素']).reset_index()

# 为了防止某些商家缺少某类超时原因导致列名缺失，确保所有可能出现的列都存在
expected_cols = ['未超时', '发货延迟导致延期', '发货/运送均延期', '运送延迟导致延期', '发货延迟（风险）']
for col in expected_cols:
    if col not in df_factors.columns:
        df_factors[col] = 0

# 4. 合并基础数据与透视数据
df_seller = pd.merge(df_seller_base, df_factors, on='seller_id', how='left').fillna(0)

# 5. 计算微观归因衍生指标
# 重新定义纯粹的总超时单数（排除“风险”和“未超时”）
df_seller['实际总超时订单数'] = (
    df_seller['发货延迟导致延期'] + 
    df_seller['发货/运送均延期'] + 
    df_seller['运送延迟导致延期']
)

# 重新定义发货导致的超时数（包含发货全锅和共同责任）
df_seller['发货导致的超时数'] = (
    df_seller['发货延迟导致延期'] + 
    df_seller['发货/运送均延期']
)

# 计算比率（0-1 之间的小数）
df_seller['超时率'] = np.where(df_seller['总订单'] > 0, df_seller['实际总超时订单数'] / df_seller['总订单'], 0)
df_seller['发货超时率'] = np.where(df_seller['总订单'] > 0, df_seller['发货导致的超时数'] / df_seller['总订单'], 0)
df_seller['纯物流超时率'] = np.where(df_seller['总订单'] > 0, df_seller['运送延迟导致延期'] / df_seller['总订单'], 0)
df_seller['潜在风险率'] = np.where(df_seller['总订单'] > 0, df_seller['发货延迟（风险）'] / df_seller['总订单'], 0)

# 超时订单中，发货因素占比
df_seller['超时订单中发货因素占比'] = np.where(
    df_seller['实际总超时订单数'] > 0, 
    df_seller['发货导致的超时数'] / df_seller['实际总超时订单数'], 
    0 
)

# 6. 整理表头顺序并重命名
df_seller = df_seller[[
    'seller_id', '总订单', '实际总超时订单数', '超时率',
    '发货导致的超时数', '发货超时率', '纯物流超时率', '潜在风险率', '超时订单中发货因素占比',
    '发货延迟导致延期', '发货/运送均延期', '运送延迟导致延期', '发货延迟（风险）', '未超时',
    '店铺平均评分', '平均支付审核通过时间_天', '平均承诺最晚发货时间_天', 
    '平均真实发货时间_天', '平均承诺运送时间_天', '平均真实运送时间_天'
]]

df_seller = df_seller.rename(columns={
    '实际总超时订单数': '超时订单数(终态)',
    '发货导致的超时数': '发货因素超时数',
    '发货延迟导致延期': '归因_纯发货导致',
    '发货/运送均延期': '归因_发货与物流共责',
    '运送延迟导致延期': '归因_纯物流导致',
    '发货延迟（风险）': '归因_发货风险(未超时)',
    '未超时': '归因_完全准时',
    '平均支付审核通过时间_天': '平均支付审核通过时间/天',
    '平均承诺最晚发货时间_天': '平均承诺最晚发货时间/天',
    '平均真实发货时间_天': '平均真实发货时间/天',
    '平均承诺运送时间_天': '平均承诺运送时间/天',
    '平均真实运送时间_天': '平均真实运送时间/天'
})

# 7. 排序：优先按总超时数排，再按发货因素超时数排
df_seller = df_seller.sort_values(
    by=['超时订单数(终态)', '发货因素超时数', '超时订单中发货因素占比'],
    ascending=[False, False, False]
).reset_index(drop=True)

# 8. 格式化输出（百分数不带%，浮点数保留三位）
# 处理百分比列
pct_cols = ['超时率', '发货超时率', '纯物流超时率', '潜在风险率', '超时订单中发货因素占比']
for col in pct_cols:
    df_seller[col] = df_seller[col].fillna(0).map(lambda x: f"{x * 100:.3f}")

# 处理浮点数时间列
num_cols = [
    '店铺平均评分', '平均支付审核通过时间/天', '平均承诺最晚发货时间/天', 
    '平均真实发货时间/天', '平均承诺运送时间/天', '平均真实运送时间/天'
]
for col in num_cols:
    df_seller[col] = df_seller[col].map(lambda x: f"{x:.3f}" if pd.notnull(x) else x)

# 9. 导出文件
target_dir = BASE_DIR / 'processed_data'
target_dir.mkdir(parents=True, exist_ok=True)
# 文件名强调订单级判断
file_path = target_dir / '商家延迟率与订单级归因表.csv'

df_seller.to_csv(file_path, index=False, encoding='utf-8-sig')

print(f"文件已成功导出至：{file_path}")
print("排序及格式化后的前 5 行数据：")
print(df_seller.head())

