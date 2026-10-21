from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).parent.parent
df = pd.read_csv(BASE_DIR / 'processed_data' / "分级后总数据.csv", encoding="utf-8-sig")
df['订单总价'] = df['价格'] + df['运费']

# ======================= 全局参数（与初步报告 A/B 实验章节一致） =======================
DAILY_LOSS_RATE = 0.018   # 每多承诺 1 天带来的转化率流失（外部文献值，待 A/B 实测替换）
MAX_EXTEND_DAYS = 7       # 单条线路可接受的最长延长天数（业务上限）
MIN_ROUTE_ORDERS = 20     # 线路进入配置表的最小订单量


def cumulative_loss_rate(days):
    """延长 days 天对应的累计转化率流失。

    采用边际递减（复利）形式而非线性外推：1 - (1 - r)^d。
    线性假设在 d 较大时会推出「延长 14 天流失 25%」这类不存在的极端值。
    """
    return 1 - (1 - DAILY_LOSS_RATE) ** days


route_base = df.groupby(['发货州', '收货州']).agg(
    总订单数=('发货州', 'size'),
    平均距离_km=('两地直线距离', 'mean'),
    线路总GMV=('订单总价', 'sum')
).reset_index()

logistics_delays = df[df['超时原因'].isin(['物流/算法', '运送延迟导致延期'])].copy()
# 去尾处理（防止极个别延误 100 天的订单把分位数拉爆）
threshold = logistics_delays['实际最终延迟时间'].quantile(0.99)
logistics_delays_trimmed = logistics_delays[logistics_delays['实际最终延迟时间'] <= threshold].copy()

# 【分位数家族】除了 P75，同时给出 P50 / P90，便于评估尾部厚度
route_logistics = logistics_delays_trimmed.groupby(['发货州', '收货州']).agg(
    物流延迟单量=('发货州', 'size'),
    延迟时长_P50=('实际最终延迟时间', lambda x: x.quantile(0.50)),
    延迟时长_P75=('实际最终延迟时间', lambda x: x.quantile(0.75)),
    延迟时长_P90=('实际最终延迟时间', lambda x: x.quantile(0.90)),
    平均延迟时长_天=('实际最终延迟时间', 'mean')
).reset_index()

route_analysis = pd.merge(route_base, route_logistics, on=['发货州', '收货州'], how='left')
fill_cols = ['物流延迟单量', '延迟时长_P50', '延迟时长_P75', '延迟时长_P90', '平均延迟时长_天']
route_analysis[fill_cols] = route_analysis[fill_cols].fillna(0)
route_analysis['物流延迟率'] = route_analysis['物流延迟单量'] / route_analysis['总订单数']

route_analysis_filtered = route_analysis[route_analysis['总订单数'] >= MIN_ROUTE_ORDERS].copy()

# 每条线路延迟订单的明细，用于计算任意延长天数下的实际挽救量
delay_map = {
    key: sub[['实际最终延迟时间', '订单总价']]
    for key, sub in logistics_delays_trimmed.groupby(['发货州', '收货州'])
}


# ================= 核心修改区：从「按 P75 取整」改为「净收益最优」 =================
# 旧逻辑只看 P75 时长，导致 SP→SP 这类「延迟率仅 5% 却有 2.4 万单」的大盘线路
# 也被无差别加上 8 天——为了救 5% 的订单，让 100% 的客户看到更长的承诺时间。
# 新逻辑显式权衡两端：
#   收益 = 延长 d 天能被救回的延迟订单 GMV（ETA 变长后这些订单不再算超时）
#   成本 = 该线路全部订单 × 累计转化率流失 × GMV（ETA 是线路级配置，改动影响全线路）
def optimize_extend_days(row):
    sub = delay_map.get((row['发货州'], row['收货州']))
    if sub is None or len(sub) == 0:
        return 0, 0, 0.0, 0.0, 0.0

    delay_arr = sub['实际最终延迟时间'].values
    price_arr = sub['订单总价'].values
    route_gmv = row['线路总GMV']

    best_d, best_net, best_prot, best_loss, best_cnt = 0, 0.0, 0.0, 0.0, 0
    for d in range(0, MAX_EXTEND_DAYS + 1):
        mask = delay_arr <= d
        protected = price_arr[mask].sum()
        lost = route_gmv * cumulative_loss_rate(d)
        net = protected - lost
        if net > best_net:
            best_net, best_d = net, d
            best_prot, best_loss, best_cnt = protected, lost, int(mask.sum())

    return best_d, best_cnt, best_prot, best_loss, best_net


opt_results = route_analysis_filtered.apply(optimize_extend_days, axis=1, result_type='expand')
opt_results.columns = ['推荐延迟天数', '预期挽救单量', '预期保护GMV', '预期流失GMV', '预期净收益GMV']
route_analysis_filtered = pd.concat([route_analysis_filtered, opt_results], axis=1)

route_analysis_filtered['预期转化流失率'] = route_analysis_filtered['推荐延迟天数'].apply(
    lambda d: cumulative_loss_rate(d) * 100
)
route_analysis_filtered['是否建议干预'] = route_analysis_filtered['推荐延迟天数'].apply(
    lambda d: '是' if d > 0 else '否'
)

# 格式整理
route_analysis_filtered['物流延迟率'] = (route_analysis_filtered['物流延迟率'] * 100).round(2)
route_analysis_filtered['平均距离_km'] = route_analysis_filtered['平均距离_km'].round(2)
for c in ['延迟时长_P50', '延迟时长_P75', '延迟时长_P90', '平均延迟时长_天']:
    route_analysis_filtered[c] = route_analysis_filtered[c].round(2)
route_analysis_filtered['预期转化流失率'] = route_analysis_filtered['预期转化流失率'].round(2)
for c in ['线路总GMV', '预期保护GMV', '预期流失GMV', '预期净收益GMV']:
    route_analysis_filtered[c] = route_analysis_filtered[c].round(2)

route_analysis_filtered = route_analysis_filtered.sort_values(
    by=['推荐延迟天数', '物流延迟率'], ascending=[False, False]
).reset_index(drop=True)

# 导出（路径与文件名保持不变）
output_dir = BASE_DIR / 'processed_data'
save_path = output_dir / '各物流线路延迟率分析.csv'
route_analysis_filtered.to_csv(save_path, index=False, encoding="utf-8-sig")

print("✅ V4版（净收益最优策略）线路配置已生成！")
print(f"\n==== 参数 ====")
print(f"  单日流失率: {DAILY_LOSS_RATE:.3%}（外部文献值，待 A/B 校准）")
print(f"  单线路延长上限: {MAX_EXTEND_DAYS} 天")
print(f"  流失模型: 1-(1-r)^d 边际递减，而非线性外推")
print("\n==== 推荐天数分布 ====")
print(route_analysis_filtered['推荐延迟天数'].value_counts().sort_index().to_string())
print(f"\n==== 建议干预线路数: {(route_analysis_filtered['是否建议干预'] == '是').sum()} / {len(route_analysis_filtered)} ====")
print(f"  预期合计保护 GMV: {route_analysis_filtered['预期保护GMV'].sum():.2f} 元")
print(f"  预期合计流失 GMV: {route_analysis_filtered['预期流失GMV'].sum():.2f} 元")
print(f"  预期合计净收益  : {route_analysis_filtered['预期净收益GMV'].sum():.2f} 元")
print("\n==== 净收益最优 Top 10 线路 ====")
print(route_analysis_filtered.head(10)[
    ['发货州', '收货州', '总订单数', '物流延迟率', '延迟时长_P75', '推荐延迟天数',
     '预期挽救单量', '预期保护GMV', '预期流失GMV', '预期净收益GMV']
].to_string(index=False))
