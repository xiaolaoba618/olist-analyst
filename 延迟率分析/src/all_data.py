import pandas as pd
from pathlib import Path

import numpy as np
# 1. 设置路径并读取数据
BASE_DIR = Path(__file__).parent.parent
df = pd.read_csv(BASE_DIR / 'data' / "订单时间段分析清洗后.csv", encoding="utf-8-sig")




# 确保 numpy 已经导入
# 定义条件和对应的结果
conditions = [
    # 情况1：最终超时了，且发货也延迟了，且发货延迟占比 > 50%
    (df['实际最终延迟时间'] > 0) & 
    (df['发货时长延迟时间'] > 0) & 
    (df['发货时长延迟时间'] / df['实际最终延迟时间'] > 0.5),
    
    # 情况2：最终超时了，且发货也延迟了，但发货延迟占比 <= 50%
    (df['实际最终延迟时间'] > 0) & 
    (df['发货时长延迟时间'] > 0) & 
    (df['发货时长延迟时间'] / df['实际最终延迟时间'] <= 0.5),
    
    # 情况3：最终超时了，但发货没延迟（发货准时或提前）
    (df['实际最终延迟时间'] > 0) & 
    (df['发货时长延迟时间'] <= 0),
    
    # 情况4：最终没超时（<=0），但发货延迟了
    (df['实际最终延迟时间'] <= 0) & 
    (df['发货时长延迟时间'] > 0)
]

# 定义对应的结果
choices = [
    "卖家发货", 
    "物流/算法", 
    "物流/算法", 
    "卖家发货（潜在风险）"
]

# 使用 np.select 生成新列，如果不满足以上任何条件，默认为 "未超时"
df['超时原因'] = np.select(conditions, choices, default="未超时")

# 检查一下生成结果的分布情况
print(df['超时原因'].value_counts(dropna=False))

target_dir = BASE_DIR / 'processed_data'
target_dir.mkdir(parents=True, exist_ok=True)
file_path = target_dir / '分级后总数据.csv'

df.to_csv(file_path, index=False, encoding='utf-8-sig')