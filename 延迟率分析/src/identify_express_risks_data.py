from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).parent.parent
df = pd.read_csv(BASE_DIR / 'processed_data' / "分级后总数据.csv", encoding="utf-8-sig")

# 1. 计算每条线路的“大盘基础数据”
route_base = df.groupby(['发货州', '收货州']).agg(
    总订单数=('发货州', 'size'),
    平均距离_km=('两地直线距离', 'mean')
).reset_index()

# 2. 筛选“物流/算法”导致的超时订单
logistics_delays = df[df['超时原因'].isin(['物流/算法', '运送延迟导致延期'])].copy()

# 3. 去尾处理（依然保留，防止极端值影响P90的计算）
threshold = logistics_delays['实际最终延迟时间'].quantile(0.99)
logistics_delays_trimmed = logistics_delays[logistics_delays['实际最终延迟时间'] <= threshold].copy()

# 4. 计算每条线路的“物流超时数据” (核心修改：均值 -> P90)
route_logistics = logistics_delays_trimmed.groupby(['发货州', '收货州']).agg(
    物流延迟单量=('发货州', 'size'),
    延迟时长_P90=('实际最终延迟时间', lambda x: x.quantile(0.90))
).reset_index()

# 5. 合并数据并计算最终指标
route_analysis = pd.merge(route_base, route_logistics, on=['发货州', '收货州'], how='left')

route_analysis['物流延迟单量'] = route_analysis['物流延迟单量'].fillna(0)
route_analysis['延迟时长_P90'] = route_analysis['延迟时长_P90'].fillna(0)
route_analysis['物流延迟率'] = route_analysis['物流延迟单量'] / route_analysis['总订单数']

# 6. 过滤小样本噪声
route_analysis_filtered = route_analysis[route_analysis['总订单数'] >= 20].copy()

# ================= 核心修改区 =================
# 计算推荐延迟天数：对 P90 向上取整，限制最小为 0，【封顶为 7 天】
route_analysis_filtered['推荐延迟天数'] = np.ceil(route_analysis_filtered['延迟时长_P90']).clip(lower=0, upper=7).astype(int)

# 格式整理
route_analysis_filtered['物流延迟率'] = (route_analysis_filtered['物流延迟率'] * 100).round(2)
route_analysis_filtered['平均距离_km'] = route_analysis_filtered['平均距离_km'].round(2)
route_analysis_filtered['延迟时长_P90'] = route_analysis_filtered['延迟时长_P90'].round(2)

# 排序
route_analysis_filtered = route_analysis_filtered.sort_values(by=['推荐延迟天数', '物流延迟率'], ascending=[False, False]).reset_index(drop=True)

# 7. 导出结果
output_dir = BASE_DIR / 'processed_data'
output_dir.mkdir(parents=True, exist_ok=True)
save_path = output_dir / '各物流线路延迟率分析.csv'
route_analysis_filtered.to_csv(save_path, index=False, encoding="utf-8-sig")

print(f"✅ 线路分析已保存至：{save_path}，最高推荐延迟天数已封顶 7 天。")
print(route_analysis_filtered[['发货州', '收货州', '物流延迟率', '延迟时长_P90', '推荐延迟天数']].head(10))