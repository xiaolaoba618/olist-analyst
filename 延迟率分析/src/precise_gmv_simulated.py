from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).parent.parent
df = pd.read_csv(BASE_DIR / 'processed_data' / "GMV分析.csv", encoding="utf-8-sig")
route_df = pd.read_csv(BASE_DIR / 'processed_data' / "各物流线路延迟率分析.csv", encoding="utf-8-sig")

# ======================= 全局参数 =======================
DAILY_LOSS_RATE = 0.018   # 单日转化率流失基准（外部文献值，待 A/B 实测替换）
DAILY_LOSS_STD = 0.005    # 该基准本身的不确定性，蒙特卡洛中作为抽样标准差
N_SIMULATIONS = 10000
PLOT_BINS = 50

# 全国平均承诺时效：「1.8%/天」这个文献值的时效锚点
T_REF = df['预计总运送周期'].mean()

# 【差评转化率】把救回单量换算成「避免的 1 星差评数」，直接度量口碑收益
STAR1_RATE_DELAYED = (df.loc[df['送达状态'] == '延迟', '订单评分'] == 1).mean()
STAR1_RATE_ONTIME = (df.loc[df['送达状态'] != '延迟', '订单评分'] == 1).mean()
STAR1_DELTA = STAR1_RATE_DELAYED - STAR1_RATE_ONTIME

route_map = route_df.set_index(['发货州', '收货州'])['推荐延迟天数'].to_dict()
df['推荐延长天数'] = df.apply(lambda x: route_map.get((x['发货州'], x['收货州']), 0), axis=1)


def eff_days_of(days, route_T):
    """把延长天数折算为「相对基准时效的等效天数」。

    客户对「多等 2 天」的敏感度取决于原承诺时效：同城 4 天的线路加 2 天是灾难，
    偏远 30 天的线路加 2 天几乎无感。文献的 1.8%/天 锚定在平均时效线路上，
    因此需按 T_REF / 线路时效 折算后再套流失模型。
    """
    if route_T is None or route_T <= 0:
        route_T = T_REF
    return days * (T_REF / route_T)


def loss_rate_of(eff_days, daily_rate=DAILY_LOSS_RATE):
    """等效天数对应的累计转化率流失，边际递减形式。"""
    return 1 - (1 - daily_rate) ** eff_days


def run_monte_carlo(target_df, mode='fixed', fixed_days=2, scope='line', label=""):
    """蒙特卡洛模拟。

    scope='line'  线路级 ETA：承诺时效按线路配置，改动后该线路【全部订单】都承担转化流失
    scope='order' 订单级 ETA：可对单个订单差异化出价时效，仅有【被干预订单】承担转化流失
    """
    total_target = len(target_df)
    if total_target == 0:
        return None, 0, 0, 0, 0

    df_calc = target_df.copy()
    df_calc['实际延长天数'] = fixed_days if mode == 'fixed' else df_calc['推荐延长天数']
    df_calc['等效延长天数'] = df_calc.apply(
        lambda x: eff_days_of(x['实际延长天数'], x['预计总运送周期']), axis=1
    )

    # ---------- 点估计 ----------
    rescued_mask = (df_calc['送达状态'] == '延迟') & (df_calc['延迟天数_天'] <= df_calc['实际延长天数'])
    rescued_cnt = int(rescued_mask.sum())
    protected_gmv = df_calc.loc[rescued_mask, '总价格'].sum()
    real_rescue_rate = rescued_cnt / total_target
    base_loss_gmv = (df_calc['总价格'] * (1 - (1 - DAILY_LOSS_RATE) ** df_calc['等效延长天数'])).sum()

    # ---------- 向量化蒙特卡洛（避免逐行 apply 拖慢 1 万次循环） ----------
    price_arr = df_calc['总价格'].to_numpy(dtype=float)
    eff_arr = df_calc['等效延长天数'].to_numpy(dtype=float)
    days_arr = df_calc['实际延长天数'].to_numpy(dtype=float)
    delay_arr = df_calc['延迟天数_天'].to_numpy(dtype=float)
    is_delay_arr = (df_calc['送达状态'] == '延迟').to_numpy()
    n = len(df_calc)

    gmv_loss_list, protected_list = [], []
    rng = np.random.default_rng(42)
    for _ in range(N_SIMULATIONS):
        # 1) 订单量波动 ±10%（有放回抽样）
        scale = rng.uniform(0.9, 1.1)
        idx = rng.integers(0, n, size=int(n * scale))
        price_s, eff_s, days_s = price_arr[idx], eff_arr[idx], days_arr[idx]
        # 2) 单日流失率本身带不确定性（截断为非负）
        daily_rate = max(0.0, rng.normal(DAILY_LOSS_RATE, DAILY_LOSS_STD))
        gmv_loss_list.append((price_s * (1 - (1 - daily_rate) ** eff_s)).sum())

        r_mask = is_delay_arr[idx] & (delay_arr[idx] <= days_s)
        protected_list.append(price_s[r_mask].sum())

    gmv_loss_array = np.array(gmv_loss_list)
    protected_array = np.array(protected_list)
    avg_loss = gmv_loss_array.mean()
    avg_protected = protected_array.mean()
    star1_saved = rescued_cnt * STAR1_DELTA

    scope_cn = '线路级 ETA（全线路订单承担流失）' if scope == 'line' else '订单级 ETA（仅干预订单承担流失）'
    print(f"\n==== {label} ====")
    print(f"  - ETA 生效口径: {scope_cn}")
    print(f"  - 计入成本的订单总量: {total_target} 单")
    print(f"  - 平均延长天数: {df_calc['实际延长天数'].mean():.2f} 天")
    print(f"  - 救回的延迟订单: {rescued_cnt} 单")
    print(f"  - 整体挽救率(救回单/计入成本订单): {real_rescue_rate * 100:.2f}%")
    print(f"  - 预期避免 1 星差评: {star1_saved:.0f} 个（1星率差 {STAR1_DELTA:.2%}）")
    print(f"  - 保护的 GMV: {protected_gmv:.2f} 元")
    print(f"  - 预期 GMV 损失: {avg_loss:.2f} 元")
    print(f"  - 95% 置信区间: [{np.percentile(gmv_loss_array, 2.5):.2f}, {np.percentile(gmv_loss_array, 97.5):.2f}]")
    print(f"  - 平均单救成本: {avg_loss / rescued_cnt if rescued_cnt else 0:.0f} 元/单")
    print(f"  - 单个差评的 GMV 代价: {avg_loss / star1_saved if star1_saved else 0:.0f} 元/个")

    return gmv_loss_array, avg_loss, protected_gmv, base_loss_gmv, rescued_cnt


# ================= 提取样本并对比三套方案 =================
logistics_orders = df[df['超时因素'] == '运送延迟导致延期'].copy()

# 方案 A：全量订单统一 +2 天（线路级 ETA，成本由 72,907 单全部承担）
res_A_arr, loss_A, prot_A, base_A, cnt_A = run_monte_carlo(
    df, mode='fixed', fixed_days=2, scope='line', label="方案A：全量统一延时 2 天"
)

# 方案 B：仅纯物流延迟订单 +2 天（订单级 ETA，成本仅由 5,021 单承担）
res_B_arr, loss_B, prot_B, base_B, cnt_B = run_monte_carlo(
    logistics_orders, mode='fixed', fixed_days=2, scope='order', label="方案B：仅纯物流延迟订单延时 2 天"
)

# 方案 C：挽救率导向的线路级阶梯（成本由【被配置线路】的全部订单承担）
# 注意分母：非干预线路推荐天数为 0、时效不变、不承担流失，若把它们计入分母会人为稀释挽救率
intervened_mask = df['推荐延长天数'] > 0
res_C_arr, loss_C, prot_C, base_C, cnt_C = run_monte_carlo(
    df[intervened_mask], mode='dynamic', scope='line', label="方案C：挽救率导向的线路级阶梯延时"
)

# ================= 结果汇总 =================
print("\n" + "=" * 60)
print("📊 【核心业务账本】三方案横向对比")
print(f"{'方案':<26}{'计入成本':>8}{'救回':>7}{'挽救率':>8}{'避免差评':>9}{'GMV流失':>12}")
for name, denom, cnt, loss in [
    ("A 全量统一+2天(线路级)", len(df), cnt_A, loss_A),
    ("B 仅物流单+2天(订单级)", len(logistics_orders), cnt_B, loss_B),
    ("C 线路级阶梯(挽救率导向)", int(intervened_mask.sum()), cnt_C, loss_C),
]:
    rate = cnt / denom * 100 if denom else 0
    print(f"{name:<26}{denom:>8}{cnt:>7}{rate:>7.2f}%{cnt * STAR1_DELTA:>9.0f}{loss:>12.0f}")
print("-" * 60)
print(f"  1 星率差（超时 {STAR1_RATE_DELAYED:.2%} vs 非延迟 {STAR1_RATE_ONTIME:.2%}）: {STAR1_DELTA:.2%}")
print(f"  时效锚点 T_REF: {T_REF:.1f} 天")
print("=" * 60)

print("\n【方案间的取舍】")
print(f"  方案C 相对方案B：多救回 {cnt_C - cnt_B} 单，"
      f"多避免 {(cnt_C - cnt_B) * STAR1_DELTA:.0f} 个 1 星差评，"
      f"多付 GMV {loss_C - loss_B:,.0f} 元")
print(f"  方案B 单救成本 {loss_B / cnt_B if cnt_B else 0:.0f} 元/单 vs "
      f"方案C 单救成本 {loss_C / cnt_C if cnt_C else 0:.0f} 元/单 "
      f"（订单级投放效率高 {(loss_C / cnt_C if cnt_C else 0) / (loss_B / cnt_B if cnt_B else 1):.1f} 倍）")
print("  注：方案C 的价值在于『当订单级 ETA 不可用时』的次优解；"
      "若具备订单级时效能力，方案B 的效率显著占优。\n")

# ================= 可视化 =================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, arr, title, color in [(axes[0], res_A_arr, '方案A：全量统一 +2 天', 'salmon'),
                              (axes[1], res_B_arr, '方案B：仅物流延迟单 +2 天', 'orange'),
                              (axes[2], res_C_arr, '方案C：挽救率导向线路级阶梯', 'lightgreen')]:
    ax.hist(arr, bins=PLOT_BINS, color=color, edgecolor='black', alpha=0.7)
    ax.axvline(arr.mean(), color='red', linestyle='dashed', linewidth=2, label=f'平均损失: {arr.mean():.0f}')
    ax.set_title(title)
    ax.legend()
plt.tight_layout()
fig_path = BASE_DIR / 'data' / 'GMV蒙特卡洛对比_三方案.png'
fig_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_path, dpi=120)
plt.close()
print(f"📈 模拟分布图已保存至：{fig_path}")

# ================= 导出线路配置表 =================
route_output = route_df.copy()
prefer_cols = ['发货州', '收货州', '平均距离_km', '总订单数', '物流延迟单量', '物流延迟率',
               '线路平均承诺时效_天', '相对时效系数', '是否入选干预池',
               '延迟时长_P50', '延迟时长_P75', '延迟时长_P90',
               '推荐延迟天数', '预期挽救单量', '线路挽救率', '预期避免1星差评数',
               '预期保护GMV', '预期流失GMV', '预期净收益GMV',
               '预期转化流失率', '单救成本_元', '是否建议干预']
existing_cols = [c for c in prefer_cols if c in route_output.columns]
route_output = route_output[existing_cols]
route_output = route_output.sort_values(by=['推荐延迟天数', '物流延迟率'], ascending=[False, False])

output_dir = BASE_DIR / '线路延迟预送时间推荐结果'
output_dir.mkdir(parents=True, exist_ok=True)
route_csv_path = output_dir / '基于P75的高挽救率线路延迟配置表.csv'
route_output.to_csv(route_csv_path, index=False, encoding="utf-8-sig")

print(f"\n✅ 线路推荐延迟配置表已成功导出至：{route_csv_path}")
print("==== 建议干预 Top 10 线路 ====")
print(route_output.head(10).to_string(index=False))
