import pandas as pd
from pathlib import Path

# 1. 设置路径
BASE_DIR = Path(__file__).parent.parent
# 读取数据
df_customer = pd.read_csv(BASE_DIR / 'processed_data' / '客户复购率分析.csv', encoding='utf-8-sig')

# 预处理：过滤掉极端的黄牛订单和运费异常，防止统计均值被极端值扭曲
price_threshold = df_customer['平均价格'].quantile(0.99)
df_clean = df_customer[df_customer['平均价格'] <= price_threshold].copy()

# ==========================================
# 2. 生成分客单价层级的统计表格
# ==========================================
# 将客户按客单价分为：低客单(<100)、中客单(100-500)、高客单(>500)
df_clean['客单分层'] = pd.cut(
    df_clean['平均价格'],
    bins=[0, 100, 500, float('inf')],
    labels=['低客单 (<100)', '中客单 (100-500)', '高客单 (>500)']
)

# 计算不同客单层的平均超时率和平均评分
value_summary = df_clean.groupby('客单分层', observed=True).agg(
    平均超时率=('超时率', 'mean'),
    平均评分=('平均评分', 'mean'),
    客户数=('customer_unique_id', 'count')
).reset_index()

# 将超时率格式化为百分比以便展示
value_summary['平均超时率'] = (value_summary['平均超时率'] * 100).round(2).astype(str) + '%'

# 生成 Markdown 格式的表格
print("==== 不同客单价层级的体验指标 ====\n")




# ==========================================
# 3. 提取高额用户名单并导出 CSV
# ==========================================
# 筛选条件：平均价格 > 500（这里基于未截断的 df_customer 筛选，确保不漏掉真正的高客单大客户）
high_value_users = df_customer[df_customer['平均价格'] > 500].copy()

# 设定输出路径：与 src 文件夹平级的 output 文件夹
output_dir = BASE_DIR / 'output'
output_dir.mkdir(parents=True, exist_ok=True)
save_path = output_dir / '高客单价用户名单.csv'

# 导出文件（按平均价格降序排列，把最有钱的客户排在最前面）
high_value_users = high_value_users.sort_values(by='平均价格', ascending=False)
high_value_users.to_csv(save_path, index=False, encoding='utf-8-sig')

print(f"\n✅ 高额用户名单已成功导出至：{save_path}")
print(f"总计高额（客单价>500）用户数：{len(high_value_users)} 人")