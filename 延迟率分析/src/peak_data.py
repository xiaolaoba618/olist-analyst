import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
# 读取原始数据
df = pd.read_csv(BASE_DIR / 'data' / "订单时间段分析清洗后.csv", encoding="utf-8-sig")

# ================= 1. 日期预处理 =================
# 确保下单时间是 datetime 格式
df['下单时间'] = pd.to_datetime(df['下单时间'])

# 提取纯日期（例如 2018-01-01），丢弃时分秒
df['日期'] = df['下单时间'].dt.date
# =================================================

# ================= 2. 按天聚合 =================
# 注意：这里的 '支付金额' 请替换为你数据表中代表销售额的实际字段名，如 '总金额'、'payment_value' 等
df_daily = df.groupby('日期').agg(
    订单总和数=('customer_unique_id', 'size'),                 # 计算当天的总行数（即订单数）
    当天营销总额=('价格', 'sum'),                          # 当天所有订单金额总和
    唯一客户数=('customer_unique_id', 'nunique'),             # 当天去重后的买家数量
    商家数量=('seller_id', 'nunique')                         # 当天有交易的去重商家数量
).reset_index()

# ================= 3. 格式化与排序 =================
# 按照日期升序排列（防止数据乱序）
df_daily = df_daily.sort_values(by='日期', ascending=True).reset_index(drop=True)

# 保留两位小数，让金额看起来更整洁
df_daily['当天营销总额'] = df_daily['当天营销总额'].map(lambda x: f"{x:.2f}")

# ================= 4. 导出 =================
target_dir = BASE_DIR / 'processed_data'
target_dir.mkdir(parents=True, exist_ok=True)
file_path = target_dir / '每日营销与用户统计.csv'

df_daily.to_csv(file_path, index=False, encoding='utf-8-sig')

print(f"文件已成功导出至：{file_path}")
print("按天聚合前 5 行数据：")
print(df_daily.head())