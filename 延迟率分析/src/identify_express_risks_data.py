from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).parent.parent
df = pd.read_csv(BASE_DIR / 'processed_data' / "分级后总数据.csv", encoding="utf-8-sig")
df['订单总价'] = df['价格'] + df['运费']

# ======================= 全局参数（与初步报告 A/B 实验章节一致） =======================
DAILY_LOSS_RATE = 0.018        # 每多承诺 1 天带来的转化率流失（外部文献值，待 A/B 实测替换）
MAX_EXTEND_DAYS = 7            # 单条线路可接受的最长延长天数（业务上限）
MIN_ROUTE_ORDERS = 20          # 线路进入配置表的最小订单量
MIN_ROUTE_DELAY_RATE = 0.12    # 线路准入阈值：物流延迟率低于此值的线路不进干预池
GMV_LOSS_BUDGET = 220000       # 干预成本预算（元），在预算内按边际效益分配延长天数

# 全国平均承诺时效，作为「1.8%/天」这个文献值的时效锚点
T_REF = df['预计总运送周期'].mean()

# 【差评转化率】超时与准时订单的 1 星率之差，用于把「救回单量」换算成「避免的差评数」。
# 直接从本平台数据统计，不引用外部系数：救回一单 = 把该单从超时态拉回准时态，
# 其 1 星概率下降的幅度就是这两个条件概率之差。
STAR1_RATE_DELAYED = (df.loc[df['最终是否超时'] == 1, '订单评分'] == 1).mean()
STAR1_RATE_ONTIME = (df.loc[df['最终是否超时'] == 0, '订单评分'] == 1).mean()
STAR1_DELTA = STAR1_RATE_DELAYED - STAR1_RATE_ONTIME


def loss_rate_of(days, route_T=None):
    """延长 days 天对应的累计转化率流失。

    两处建模选择：
    1) 边际递减（复利）：1-(1-r)^d，而非线性外推。线性会在 d=14 时推出「流失 25%」这类不存在的数。
    2) 相对时效折算：文献的 1.8%/天 锚定在平均时效（约 23 天）的线路上。客户对「多等 2 天」的
       敏感度取决于原承诺时效——同城 4 天的线路加 2 天是灾难，偏远 30 天的线路加 2 天几乎无感。
       因此把延长天数折算成「相对基准时效的等效天数」：等效天数 = d × (T_REF / 线路时效)。
    """
    if route_T is None or route_T <= 0:
        route_T = T_REF
    eff_days = days * (T_REF / route_T)
    return 1 - (1 - DAILY_LOSS_RATE) ** eff_days


route_base = df.groupby(['发货州', '收货州']).agg(
    总订单数=('发货州', 'size'),
    平均距离_km=('两地直线距离', 'mean'),
    线路总GMV=('订单总价', 'sum'),
    线路平均承诺时效_天=('预计总运送周期', 'mean')
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
route_analysis_filtered['相对时效系数'] = route_analysis_filtered['线路平均承诺时效_天'].apply(
    lambda T: round(T_REF / T, 3) if T > 0 else 1.0
)

# 每条线路物流延迟订单的明细，用于计算任意延长天数下的实际挽救量
delay_map = {
    key: sub['实际最终延迟时间'].values
    for key, sub in logistics_delays_trimmed.groupby(['发货州', '收货州'])
}


# ============ 核心改动（V5）：从「净收益最优」改为「挽救率导向 + 预算约束」 ============
# 上一版（V4）以「净收益 = 保护GMV - 流失GMV」最大为目标，结果过于保守：
# 线路级 ETA 下，救一单的成本（约 230 元）高于一单本身的 GMV（约 160 元），
# 于是最优解大量落在 d=0，方案 C 几乎没救回任何延迟订单。
#
# 但「整体挽救率 = 救回单 / 计入成本的订单」直接决定差评量与平台口碑，不能唯 GMV 论。
# 因此本版把目标函数换成：在成本预算内，最大化整体挽救率。
#
# 两个关键机制：
#   ① 准入筛选：物流延迟率 < 12% 的线路不进干预池。
#      挽救率的天花板 = 干预池的整体延迟率，把大盘低延迟线路（如 SP→SP）放进来只会稀释分母。
#   ② 边际贪心：把每條線路的延长期拆成 1→7 天的边际步骤，按「每救一单的成本」升序投放，
#      直到预算耗尽。天然跳过「再加一天也救不回几单」的无效区间。
def build_marginal_steps(route_df):
    """把每条线路的延长期拆成边际步骤（每天一步）。"""
    steps = []
    for idx, row in route_df.iterrows():
        key = (row['发货州'], row['收货州'])
        delay_arr = delay_map.get(key)
        if delay_arr is None:
            continue
        route_gmv = row['线路总GMV']
        route_T = row['线路平均承诺时效_天']
        prev_rescued = 0
        prev_loss = 0.0
        for d in range(1, MAX_EXTEND_DAYS + 1):
            rescued = int((delay_arr <= d).sum())
            inc = rescued - prev_rescued
            loss = route_gmv * loss_rate_of(d, route_T)
            inc_loss = loss - prev_loss
            if inc > 0 and inc_loss > 0:
                steps.append({
                    'idx': idx, 'days': d,
                    '新增挽救单量': inc, '新增成本': inc_loss,
                    '单救成本': inc_loss / inc
                })
            prev_rescued, prev_loss = rescued, loss
    return pd.DataFrame(steps)


# 干预池：延迟率达到阈值的线路
route_analysis_filtered['是否入选干预池'] = (
    route_analysis_filtered['物流延迟率'] >= MIN_ROUTE_DELAY_RATE
).map({True: '是', False: '否'})
pool = route_analysis_filtered[route_analysis_filtered['物流延迟率'] >= MIN_ROUTE_DELAY_RATE].copy()

steps_df = build_marginal_steps(pool).sort_values('单救成本').reset_index(drop=True)

# ---------------- 边际贪心分配 ----------------
# 需多轮扫描：某条线路第 d 天的边际步可能因「天数连续性」被暂时跳过，
# 一旦第 d-1 天在后续轮次被选中，第 d 天必须重新获得被选中的机会，否则预算永远用不完。
chosen_days = {idx: 0 for idx in pool.index}
spent = 0.0
improved = True
while improved:
    improved = False
    for _, st in steps_df.iterrows():
        idx = st['idx']
        # 保证天数连续：只有当前已选天数 +1 的边际步才允许投放
        if st['days'] != chosen_days[idx] + 1:
            continue
        if spent + st['新增成本'] > GMV_LOSS_BUDGET:
            continue
        spent += st['新增成本']
        chosen_days[idx] = int(st['days'])
        improved = True


def evaluate_route(row):
    """回算单条线路在选定延长期下的各项预期指标。"""
    idx = row.name
    d = chosen_days.get(idx, 0)
    key = (row['发货州'], row['收货州'])
    delay_arr = delay_map.get(key)
    if d == 0 or delay_arr is None:
        return pd.Series([0, 0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rescued = int((delay_arr <= d).sum())
    loss = row['线路总GMV'] * loss_rate_of(d, row['线路平均承诺时效_天'])
    # 被救回的延迟订单 GMV（体验保护量，非真实增收）
    protected = row['线路总GMV'] * (rescued / row['总订单数']) if row['总订单数'] > 0 else 0.0
    unit_cost = loss / rescued if rescued > 0 else 0.0
    return pd.Series([
        d, rescued, protected, loss, protected - loss,
        loss_rate_of(d, row['线路平均承诺时效_天']) * 100,
        unit_cost
    ])


res_cols = ['推荐延迟天数', '预期挽救单量', '预期保护GMV', '预期流失GMV',
            '预期净收益GMV', '预期转化流失率', '单救成本_元']
route_analysis_filtered[res_cols] = route_analysis_filtered.apply(evaluate_route, axis=1)
for c in ['推荐延迟天数', '预期挽救单量']:
    route_analysis_filtered[c] = route_analysis_filtered[c].astype(int)

route_analysis_filtered['线路挽救率'] = (
    route_analysis_filtered['预期挽救单量'] / route_analysis_filtered['总订单数'] * 100
)
# 把救回单量折算为「避免的 1 星差评数」——这是差评量与平台口碑的直接度量
route_analysis_filtered['预期避免1星差评数'] = (
    route_analysis_filtered['预期挽救单量'] * STAR1_DELTA
).round(1)
route_analysis_filtered['是否建议干预'] = route_analysis_filtered['推荐延迟天数'].apply(
    lambda d: '是' if d > 0 else '否'
)

# 格式整理
route_analysis_filtered['物流延迟率'] = (route_analysis_filtered['物流延迟率'] * 100).round(2)
route_analysis_filtered['平均距离_km'] = route_analysis_filtered['平均距离_km'].round(2)
route_analysis_filtered['线路平均承诺时效_天'] = route_analysis_filtered['线路平均承诺时效_天'].round(2)
for c in ['延迟时长_P50', '延迟时长_P75', '延迟时长_P90', '平均延迟时长_天']:
    route_analysis_filtered[c] = route_analysis_filtered[c].round(2)
for c in ['线路挽救率', '预期转化流失率']:
    route_analysis_filtered[c] = route_analysis_filtered[c].round(2)
for c in ['线路总GMV', '预期保护GMV', '预期流失GMV', '预期净收益GMV', '单救成本_元']:
    route_analysis_filtered[c] = route_analysis_filtered[c].round(2)

route_analysis_filtered = route_analysis_filtered.sort_values(
    by=['推荐延迟天数', '物流延迟率'], ascending=[False, False]
).reset_index(drop=True)

# 导出（路径与文件名保持不变）
output_dir = BASE_DIR / 'processed_data'
save_path = output_dir / '各物流线路延迟率分析.csv'
route_analysis_filtered.to_csv(save_path, index=False, encoding="utf-8-sig")

# ---------------- 汇总输出 ----------------
act = route_analysis_filtered[route_analysis_filtered['推荐延迟天数'] > 0]
denominator = act['总订单数'].sum()
rescued = act['预期挽救单量'].sum()
cost = act['预期流失GMV'].sum()
saved_rate = rescued / denominator * 100 if denominator else 0.0

# 干预池天花板：池内线路全部拉满 7 天时的挽救率（衡量本方案离上限还有多远）
pool_ceil_rescued = 0
for _, row in pool.iterrows():
    da = delay_map.get((row['发货州'], row['收货州']))
    if da is not None:
        pool_ceil_rescued += int((da <= MAX_EXTEND_DAYS).sum())
pool_denominator = pool['总订单数'].sum()
ceil_rate = pool_ceil_rescued / pool_denominator * 100 if pool_denominator else 0.0

print("✅ V5版（挽救率导向 + 预算约束）线路配置已生成！")
print("\n==== 参数 ====")
print(f"  单日流失率: {DAILY_LOSS_RATE:.3%}（外部文献值，待 A/B 校准）")
print(f"  时效锚点 T_REF: {T_REF:.1f} 天（全国平均承诺时效，用于相对时效折算）")
print(f"  单线路延长上限: {MAX_EXTEND_DAYS} 天")
print(f"  线路准入延迟率阈值: {MIN_ROUTE_DELAY_RATE:.0%}")
print(f"  成本预算: {GMV_LOSS_BUDGET / 10000:.1f} 万元")
print(f"  流失模型: 1-(1-r)^(d×T_REF/T线路)，边际递减 + 相对时效折算")

print("\n==== 干预池 ====")
print(f"  候选线路（≥{MIN_ROUTE_ORDERS}单）: {len(route_analysis_filtered)} 条")
print(f"  入选干预池（延迟率≥{MIN_ROUTE_DELAY_RATE:.0%}）: {len(pool)} 条 | 覆盖订单 {pool_denominator:,} 单")
print(f"  实际配置到延长天数: {len(act)} 条 | 覆盖订单 {denominator:,} 单")

print("\n==== 挽救效果 ====")
print(f"  预期救回延迟订单: {rescued:,} 单")
print(f"  计入成本的订单（分母）: {denominator:,} 单")
print(f"  整体挽救率: {saved_rate:.2f}%")
print(f"  预期 GMV 流失成本: {cost:,.0f} 元（预算 {GMV_LOSS_BUDGET / 10000:.1f} 万，"
      f"执行率 {cost / GMV_LOSS_BUDGET * 100:.1f}%）")
print(f"  平均单救成本: {cost / rescued if rescued else 0:.0f} 元/单")
print(f"  干预池理论天花板（全部拉满 {MAX_EXTEND_DAYS} 天）: {pool_ceil_rescued:,} 单 / {ceil_rate:.2f}%")
print(f"  已达天花板的: {saved_rate / ceil_rate * 100 if ceil_rate else 0:.1f}%")

star1_saved = rescued * STAR1_DELTA
print("\n==== 口碑收益（1 星差评口径）====")
print(f"  超时订单 1 星率: {STAR1_RATE_DELAYED:.2%} | 准时订单 1 星率: {STAR1_RATE_ONTIME:.2%} "
      f"| 差值: {STAR1_DELTA:.2%}")
print(f"  预期避免 1 星差评: {star1_saved:.0f} 个")
print(f"  单个差评的 GMV 代价: {cost / star1_saved if star1_saved else 0:.0f} 元/个")

print("\n==== 推荐天数分布 ====")
print(route_analysis_filtered['推荐延迟天数'].value_counts().sort_index().to_string())

print("\n==== Top 10 配置线路 ====")
print(act.head(10)[
    ['发货州', '收货州', '总订单数', '物流延迟率', '延迟时长_P75', '推荐延迟天数',
     '预期挽救单量', '线路挽救率', '预期流失GMV', '单救成本_元']
].to_string(index=False))
