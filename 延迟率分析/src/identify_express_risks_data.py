from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).parent.parent
df = pd.read_csv(BASE_DIR / 'processed_data' / "分级后总数据.csv", encoding="utf-8-sig")

route_base = df.groupby(['发货州', '收货州']).agg(
    总订单数=('发货州', 'size'),
    平均距离_km=('两地直线距离', 'mean')
).reset_index()

logistics_delays = df[df['超时原因'].isin(['物流/算法', '运送延迟导致延期'])].copy()
# 去尾处理（防止极个别延误 100 天的订单把分位数拉爆）
threshold = logistics_delays['实际最终延迟时间'].quantile(0.99)
logistics_delays_trimmed = logistics_delays[logistics_delays['实际最终延迟时间'] <= threshold].copy()

# 【核心修改】：计算该线路延误订单的 P75 分位数
route_logistics = logistics_delays_trimmed.groupby(['发货州', '收货州']).agg(
    物流延迟单量=('发货州', 'size'),
    延迟时长_P75=('实际最终延迟时间', lambda x: x.quantile(0.75))
).reset_index()

route_analysis = pd.merge(route_base, route_logistics, on=['发货州', '收货州'], how='left')
route_analysis['物流延迟单量'] = route_analysis['物流延迟单量'].fillna(0)
route_analysis['延迟时长_P75'] = route_analysis['延迟时长_P75'].fillna(0)
route_analysis['物流延迟率'] = route_analysis['物流延迟单量'] / route_analysis['总订单数']

route_analysis_filtered = route_analysis[route_analysis['总订单数'] >= 20].copy()


# ================= 核心修改区：绝不妥协的阶梯策略 =================
def calculate_aggressive_steps(row):
    delay_rate = row['物流延迟率']
    p75_days = row['延迟时长_P75']

    # 规则1：延迟率极低（<3%），且 P75 小于1天，不延长
    if delay_rate < 0.03 and p75_days < 1:
        return 0

    # 规则2：对于有延迟的线路，直接按 P75 向上取整
    # 比如 P75 是 2.1 天，取整为 3 天；P75 是 4.8 天，取整为 5 天
    recommended = int(np.ceil(p75_days))

    # 规则3：强制兜底，最少延 1 天，最多延 7 天（不再有 -1 放弃策略）
    return max(1, min(recommended, 7))


route_analysis_filtered['推荐延迟天数'] = route_analysis_filtered.apply(calculate_aggressive_steps, axis=1)

# 格式整理
route_analysis_filtered['物流延迟率'] = (route_analysis_filtered['物流延迟率'] * 100).round(2)
route_analysis_filtered['平均距离_km'] = route_analysis_filtered['平均距离_km'].round(2)
route_analysis_filtered['延迟时长_P75'] = route_analysis_filtered['延迟时长_P75'].round(2)

route_analysis_filtered = route_analysis_filtered.sort_values(by=['推荐延迟天数', '物流延迟率'],
                                                              ascending=[False, False]).reset_index(drop=True)

# 导出为 V3 版本
output_dir = BASE_DIR / 'processed_data'
save_path = output_dir / '各物流线路延迟率分析.csv'
route_analysis_filtered.to_csv(save_path, index=False, encoding="utf-8-sig")

print("✅ V3版（高挽救率策略）配置已生成！")
print("\n==== 推荐天数分布 (大量线路会集中在 3-6 天) ====")
print(route_analysis_filtered['推荐延迟天数'].value_counts().sort_index(ascending=False))