import pandas as pd
from pathlib import Path

# 1. 设置路径并读取数据
BASE_DIR = Path(__file__).parent.parent
df = pd.read_csv(BASE_DIR / 'data' / "订单时间段分析清洗后.csv", encoding="utf-8-sig")
df_s = pd.read_csv(BASE_DIR / '商家层面分析结果' / "seller_risk_rule_full.csv", encoding="utf-8-sig")
df_g = pd.read_csv(BASE_DIR / '商品层面分析结果' / "category_risk_rule_full.csv", encoding="utf-8-sig")

# 1. 商家表：改名后只保留需要合并的列
df_s = df_s[['seller_id', '风险等级', '超时归因']].rename(
    columns={
        '风险等级': '商家风险等级',
        '超时归因': '商家超时归因'
    }
)

# 2. 商品表：风险等级 → 商品风险等级
df_g = df_g[['商品类型', '风险等级']].rename(
    columns={'风险等级': '商品风险等级'}
)

# 3. 两次 left join 到原表
df = df.merge(df_s, on='seller_id', how='left')
df = df.merge(df_g, on='商品类型', how='left')

target_dir = BASE_DIR / 'processed_data'
target_dir.mkdir(parents=True, exist_ok=True)
file_path = target_dir / '分级后总数据.csv'

df.to_csv(file_path, index=False, encoding='utf-8-sig')

