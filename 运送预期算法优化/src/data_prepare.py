# %%
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, average_precision_score
from config import CONFIG

# 设置中文字体，防止图表中文乱码 (Windows用SimHei，Mac用Arial Unicode MS)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = CONFIG['base_dir']
print(BASE_DIR)
# %%
df = pd.read_csv(BASE_DIR /'processed_data' / "LightGBM.csv", encoding="utf-8-sig")
import pandas as pd
rename_dict = {
    "下单时间": "order_time",
    "seller_id": "seller_id",
    "商品类型": "product_type",
    "付款方式": "payment_method",
    "发货城市": "seller_city",
    "发货州": "seller_state",
    "收货城市": "customer_city",
    "收货州": "customer_state",
    '预期总周期': 'estimated_delivery_time',
    "总运送周期": "total_delivery_time",
    "对数质量": "log_weight",
    "对数体积": "log_volume",
    "对数价格": "log_price",
    "对数运费": "log_freight",
    "对数距离": "log_distance",
    "运费占比": "freight_ratio",
    "单位距离运费": "freight_per_km",
    "对数运送周期": "log_delivery_time",
    "星期_周一始": "day_of_week",
    "是否周末": "is_weekend",
    "平台_7天单量": "platform_7d_orders",
    "平台_7天平均审核时长": "platform_7d_avg_approval_time",
    "卖家_7天单量": "seller_7d_orders",
    "卖家_7天平均发货时长": "seller_7d_avg_shipping_time",
    "卖家_7天超时率": "seller_7d_late_rate",
    "卖家_前7天单量": "seller_prev_7d_orders",
    "卖家_单量增幅": "seller_order_growth",
    "卖家_历史最高7天单量": "seller_max_7d_orders",
    "卖家_产能负荷率": "seller_capacity_load",
    "是否黑五月份": "is_black_friday_month",
    "是否黑五当天": "is_black_friday_day",
    "是否黑五当周": "is_black_friday_week",
    "是否为大件": "is_large",
    "路线P90_总历史": "route_p90_total",
    "路线超时率_总历史": "route_late_rate_total",
    "路线P90_近期90天": "route_p90_90d",
    "路线超时率_近期90天": "route_late_rate_90d",
    "是否为偏远州": "is_remote",
    "是否为国际订单": "is_international"
}


print(len(rename_dict))
# 3. 执行重命名
df = df.rename(columns=rename_dict)

# 4. 基础格式转换
df['order_time'] = pd.to_datetime(df['order_time'], format='mixed')

cat_cols = ['product_type', 'payment_method', 'seller_state', 'customer_state']
for col in cat_cols:
    df[col] = df[col].astype('category')

import numpy as np

# 1. 处理可能存在的无穷大（inf 替换为缺失值）
df = df.replace([np.inf, -np.inf], np.nan)

# 2. 安全地将 0/1 标签转换为整数
binary_cols = ['is_weekend', 'is_black_friday_month', 'is_black_friday_day', 
               'is_black_friday_week', 'is_large', 'is_remote', 'is_international']

for col in binary_cols:
    if col in df.columns:
        # 先用 0 填补标签列中的缺失值（逻辑上，缺失未知默认按 0 处理）
        df[col] = df[col].fillna(0)
        # 然后再转为 int
        df[col] = df[col].astype(int)

# 3. 检查其他连续特征是否存在缺失值或无穷大（LightGBM 可以处理 NaN，但不能处理 inf）
# 找出所有数值型列
numeric_cols = df.select_dtypes(include=[np.number]).columns
inf_cols = [col for col in numeric_cols if np.isinf(df[col]).any()]

if len(inf_cols) > 0:
    print(f"⚠️ 警告：以下列包含无穷大，已替换为 NaN：{inf_cols}")
    df[inf_cols] = df[inf_cols].replace([np.inf, -np.inf], np.nan)


# %%
df.to_csv(BASE_DIR /'processed_data' / "LightGBM.csv", index=False, encoding="utf-8-sig")
# %%
# 计算差值
diff = df["total_delivery_time"] - df["estimated_delivery_time"]

# 差值 >0 的样本数量
cnt_pos = (diff > 0).sum()

# 总行数
n_total = len(df)

# 占比
ratio = cnt_pos / n_total

print(f"差值>0样本数: {cnt_pos}, 总样本: {n_total}, 占比: {ratio:.4f}")
