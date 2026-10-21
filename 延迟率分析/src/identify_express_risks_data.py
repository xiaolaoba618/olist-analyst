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
GMV_LOSS_BUDGET = 300000       # 干预成本预算（元），在预算内按边际效益分配延长天数

# 全国平均承诺时效，作为「1.8%/天」这个文献值的时效锚点
T_REF = df['预计总运送周期'].mean()


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


# ============ 分析主体：只保留「物流/算法」原因导致的超时订单 ============
# 超时原因共四类：未超时 61,223 / 物流-算法 5,830 / 卖家发货（潜在风险）4,826 / 卖家发货 1,028。
# 本次要治理的是物流与时效预测造成的问题，卖家发货责任不在延长 ETA 的解决范围内（那属于商家治理），
# 因此后续所有线路统计、挽救量、挽救率，全部在这 5,830 单的口径内计算。
logistics_delays = df[df['超时原因'] == '物流/算法'].copy()
TOTAL_LOGISTICS_DELAYED = len(logistics_delays)

# 分位数展示用的去尾子集（防止极个别延误 100 天的订单把 P90 拉爆）。
# 注意：去尾只影响分位数展示，实际挽救量仍按全量 5,830 单计算，否则会低估可挽救空间。
threshold = logistics_delays['实际最终延迟时间'].quantile(0.99)
logistics_delays_trimmed = logistics_delays[logistics_delays['实际最终延迟时间'] <= threshold].copy()

# 【差评转化率】超时与准时订单的 1 星率之差，用于把「救回单量」换算成「避免的差评数」
STAR1_RATE_DELAYED = (df.loc[df['最终是否超时'] == 1, '订单评分'] == 1).mean()
STAR1_RATE_ONTIME = (df.loc[df['最终是否超时'] == 0, '订单评分'] == 1).mean()
STAR1_DELTA = STAR1_RATE_DELAYED - STAR1_RATE_ONTIME

route_base = df.groupby(['发货州', '收货州']).agg(
    总订单数=('发货州', 'size'),
    平均距离_km=('两地直线距离', 'mean'),
    线路总GMV=('订单总价', 'sum'),
    线路平均承诺时效_天=('预计总运送周期', 'mean')
).reset_index()

# 【分位数家族】同时给出 P50 / P75 / P90，便于评估尾部厚度
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

# 每条线路物流延迟订单的明细（延迟时长 + 订单总价），用于计算任意延长天数下的实际挽救量与保护 GMV
# 用 (发货州, 收货州) 元组做键而非 DataFrame 索引：后续会对线路表排序，索引会变，
# 用索引做键会让「步骤表」与「线路表」的对应关系错位。
delay_map = {
    key: sub[['实际最终延迟时间', '订单总价']]
    for key, sub in logistics_delays.groupby(['发货州', '收货州'])
}


def route_key(row):
    return (row['发货州'], row['收货州'])


# ============ 核心（V6）：预算约束下最大化「物流延迟订单挽救率」 ============
# 挽救率 = 救回的物流/算法超时单 / 物流/算法超时单总数（5,830，全局固定分母）。
# 分母固定，所以「最大化挽救率」等价于「最大化救回单数」，目标函数因此非常干净：
# 在成本预算内，把每一分钱都投到「多救一单成本最低」的地方。
#
# 上一版（V5）额外加了「延迟率≥12% 才准入」的限制，实测是负优化：
# 同样的 22 万预算，有准入只救回 933 单，去掉准入能救回 1,099 单，多 18%。
# 准入的本意是控制分母，但本口径下分母是全体物流延迟单、与干预范围无关，
# 人为砍掉线路只会白白损失可挽救的订单。故本版移除该阈值。
def build_marginal_steps(route_df):
    """把每条线路的延长期拆成边际步骤（每天一步）。"""
    steps = []
    for _, row in route_df.iterrows():
        key = route_key(row)
        sub = delay_map.get(key)
        if sub is None:
            continue
        delay_arr = sub['实际最终延迟时间'].values
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
                    'route': key, 'days': d,
                    '新增挽救单量': inc, '新增成本': inc_loss,
                    '单救成本': inc_loss / inc
                })
            prev_rescued, prev_loss = rescued, loss
    return pd.DataFrame(steps)


steps_df = build_marginal_steps(route_analysis_filtered).sort_values('单救成本').reset_index(drop=True)


def allocate(budget):
    """在给定预算下按「单救成本」升序做边际贪心，返回 {线路: 延长天数} 与花费。

    需多轮扫描：某条线路第 d 天的边际步可能因「天数连续性」被暂时跳过，
    一旦第 d-1 天在后续轮次被选中，第 d 天必须重新获得被选中的机会，否则预算永远用不完。
    """
    chosen = {key: 0 for key in delay_map}
    spent = 0.0
    improved = True
    while improved:
        improved = False
        for st in steps_df.itertuples(index=False):
            if st.days != chosen[st.route] + 1:
                continue
            if spent + st.新增成本 > budget:
                continue
            spent += st.新增成本
            chosen[st.route] = int(st.days)
            improved = True
    return chosen, spent


chosen_days, spent = allocate(GMV_LOSS_BUDGET)


def evaluate_route(row):
    """回算单条线路在选定延长期下的各项预期指标。"""
    key = route_key(row)
    d = chosen_days.get(key, 0)
    sub = delay_map.get(key)
    if d == 0 or sub is None:
        return pd.Series([0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rescued_mask = sub['实际最终延迟时间'] <= d
    rescued = int(rescued_mask.sum())
    # 保护 GMV = 被救回订单的实际 GMV（体验保护量，非真实增收）
    protected = float(sub.loc[rescued_mask, '订单总价'].sum())
    loss = row['线路总GMV'] * loss_rate_of(d, row['线路平均承诺时效_天'])
    # 线路级挽救率：该线路物流延迟单中被救回的比例
    route_rate = rescued / len(sub) * 100 if len(sub) else 0.0
    unit_cost = loss / rescued if rescued > 0 else 0.0
    return pd.Series([
        d, rescued, route_rate, loss,
        loss_rate_of(d, row['线路平均承诺时效_天']) * 100, unit_cost,
        rescued * STAR1_DELTA, protected
    ])


res_cols = ['推荐延迟天数', '预期挽救单量', '线路挽救率', '预期流失GMV',
            '预期转化流失率', '单救成本_元', '预期避免1星差评数', '预期保护GMV']
route_analysis_filtered[res_cols] = route_analysis_filtered.apply(evaluate_route, axis=1)
for c in ['推荐延迟天数', '预期挽救单量']:
    route_analysis_filtered[c] = route_analysis_filtered[c].astype(int)

route_analysis_filtered['预期净收益GMV'] = (
    route_analysis_filtered['预期保护GMV'] - route_analysis_filtered['预期流失GMV']
).round(2)
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
route_analysis_filtered['预期避免1星差评数'] = route_analysis_filtered['预期避免1星差评数'].round(1)
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
rescued = act['预期挽救单量'].sum()
cost = act['预期流失GMV'].sum()
global_rate = rescued / TOTAL_LOGISTICS_DELAYED * 100
star1_saved = rescued * STAR1_DELTA

# 理论上限：所有线路拉满 7 天
ceil_rescued = sum(int((sub['实际最终延迟时间'] <= MAX_EXTEND_DAYS).sum())
                   for sub in delay_map.values())
ceil_rate = ceil_rescued / TOTAL_LOGISTICS_DELAYED * 100

print("✅ V6版（物流延迟订单挽救率导向 + 预算约束）线路配置已生成！")
print("\n==== 分析口径 ====")
print(f"  分析对象: 超时原因 = 物流/算法，共 {TOTAL_LOGISTICS_DELAYED:,} 单（全局固定分母）")
print(f"  挽救率定义: 救回的物流延迟单 / {TOTAL_LOGISTICS_DELAYED:,} 单")
print(f"  去尾子集（仅用于分位数展示）: {len(logistics_delays_trimmed):,} 单")

print("\n==== 参数 ====")
print(f"  单日流失率: {DAILY_LOSS_RATE:.3%}（外部文献值，待 A/B 校准）")
print(f"  时效锚点 T_REF: {T_REF:.1f} 天")
print(f"  单线路延长上限: {MAX_EXTEND_DAYS} 天")
print(f"  成本预算: {GMV_LOSS_BUDGET / 10000:.0f} 万元")
print(f"  流失模型: 1-(1-r)^(d×T_REF/T线路)，边际递减 + 相对时效折算")

print("\n==== 挽救效果 ====")
print(f"  配置到延长天数的线路: {len(act)} 条 / 候选 {len(route_analysis_filtered)} 条")
print(f"  预期救回物流延迟订单: {rescued:,} 单")
print(f"  整体挽救率（救回 / {TOTAL_LOGISTICS_DELAYED:,}）: {global_rate:.2f}%")
print(f"  预期 GMV 流失成本: {cost:,.0f} 元（预算 {GMV_LOSS_BUDGET / 10000:.0f} 万，"
      f"执行率 {cost / GMV_LOSS_BUDGET * 100:.1f}%）")
print(f"  平均单救成本: {cost / rescued if rescued else 0:.0f} 元/单")
print(f"  理论上限（全部拉满 {MAX_EXTEND_DAYS} 天）: {ceil_rescued:,} 单 / {ceil_rate:.2f}%")
print(f"  已达上限的: {global_rate / ceil_rate * 100 if ceil_rate else 0:.1f}%")

print("\n==== 口碑收益 ====")
print(f"  超时单 1 星率 {STAR1_RATE_DELAYED:.2%} vs 准时单 {STAR1_RATE_ONTIME:.2%} "
      f"| 差值 {STAR1_DELTA:.2%}")
print(f"  预期避免 1 星差评: {star1_saved:.0f} 个")
print(f"  单个差评的 GMV 代价: {cost / star1_saved if star1_saved else 0:.0f} 元/个")

print("\n==== 预算敏感性（决策用）====")
print(f"{'预算(万)':>8}{'救回单':>8}{'挽救率':>9}{'成本(万)':>10}{'元/单':>8}")
for B in [5, 10, 15, 22, 30, 50, 80, 120]:
    ch, sp = allocate(B * 10000)
    num = 0
    for key, d in ch.items():
        if d > 0:
            num += int((delay_map[key]['实际最终延迟时间'] <= d).sum())
    flag = ' ← 当前设定' if B == GMV_LOSS_BUDGET // 10000 else ''
    print(f"{B:>8}{num:>8}{num / TOTAL_LOGISTICS_DELAYED * 100:>8.2f}%{sp / 10000:>10.1f}"
          f"{sp / num if num else 0:>8.0f}{flag}")

print("\n==== 推荐天数分布 ====")
print(route_analysis_filtered['推荐延迟天数'].value_counts().sort_index().to_string())

print("\n==== Top 10 配置线路 ====")
print(act.head(10)[
    ['发货州', '收货州', '总订单数', '物流延迟单量', '物流延迟率', '延迟时长_P75',
     '推荐延迟天数', '预期挽救单量', '线路挽救率', '单救成本_元']
].to_string(index=False))
