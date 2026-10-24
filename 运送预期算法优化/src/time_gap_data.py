
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# 设置中文字体，防止图表中文乱码 (Windows用SimHei，Mac用Arial Unicode MS)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).parent.parent.parent
print(BASE_DIR)


df = pd.read_csv(BASE_DIR /'运送预期算法优化' / 'data' / "订单数据.csv", encoding="utf-8-sig")
df_delay = pd.read_csv(BASE_DIR /'延迟率分析' / '线路延迟预送时间推荐结果' / "基于P75的高挽救率线路延迟配置表.csv", encoding="utf-8-sig")


# 1. 左连接合并
df_merged = pd.merge(
    df, 
    df_delay[['发货州', '收货州', '推荐延迟天数']], 
    on=['发货州', '收货州'], 
    how='left'
)

# 2. 处理可能产生的 NaN（如果某些州组合在 df_delay 中不存在，左连接后会是 NaN）
# 假设没有匹配到的就当做推荐延迟为 0
df_merged['推荐延迟天数'] = df_merged['推荐延迟天数'].fillna(0)

# 3. 相加：实际延迟 + 推荐延迟
df_merged['送达时间差_优化后'] = df_merged['送达时间差'] + df_merged['推荐延迟天数']

df_merged.to_csv(BASE_DIR /'运送预期算法优化' / 'data' / "订单数据_延迟优化后.csv", index=False, encoding="utf-8-sig")

