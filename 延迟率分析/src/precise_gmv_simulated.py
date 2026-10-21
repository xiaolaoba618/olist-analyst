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

route_map = route_df.set_index(['发货州', '收货州'])['推荐延迟天数'].to_dict()
df['推荐延长天数'] = df.apply(lambda x: route_map.get((x['发货州'], x['收货州']), 0), axis=1)


def loss_rate_of(days, daily_rate=DAILY_LOSS_RATE):
    """延长 days 天的累计转化率流失，边际递减形式。"""
    return 1 - (1 - daily_rate) ** days


def run_monte_carlo(target_df, mode='fixed', fixed_days=2, scope='line', label=""):
    """蒙特卡洛模拟。

    scope='line'  线路级 ETA：承诺时效按线路配置，改动后该线路【全部订单】都承担转化流失
    scope='order' 订单级 ETA：可对单个订单差异化出价时效，仅有【被干预订单】承担转化流失
    """
    total_target = len(target_df)
    if total_target == 0:
        return None, 0, 0, 0

    df_calc = target_df.copy()
    df_calc['实际延长天数'] = fixed_days if mode == 'fixed' else df_calc['推荐延长天数']

    # ---------- 点估计 ----------
    rescued_mask = (df_calc['送达状态'] == '延迟') & (df_calc['延迟天数_天'] <= df_calc['实际延长天数'])
    protected_gmv = df_calc.loc[rescued_mask, '总价格'].sum()
    real_rescue_rate = rescued_mask.sum() / total_target
    base_loss_gmv = (df_calc['总价格'] * (1 - (1 - DAILY_LOSS_RATE) ** df_calc['实际延长天数'])).sum()

    # ---------- 向量化蒙特卡洛（避免逐行 apply 拖慢 1 万次循环） ----------
    price_arr = df_calc['总价格'].to_numpy(dtype=float)
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
        price_s, days_s = price_arr[idx], days_arr[idx]
        # 2) 单日流失率本身带不确定性（截断为非负）
        daily_rate = max(0.0, rng.normal(DAILY_LOSS_RATE, DAILY_LOSS_STD))
        gmv_loss_list.append((price_s * (1 - (1 - daily_rate) ** days_s)).sum())

        r_mask = is_delay_arr[idx] & (delay_arr[idx] <= days_s)
        protected_list.append(price_s[r_mask].sum())

    gmv_loss_array = np.array(gmv_loss_list)
    protected_array = np.array(protected_list)
    avg_loss = gmv_loss_array.mean()
    avg_protected = protected_array.mean()

    scope_cn = '线路级 ETA（全线路订单承担流失）' if scope == 'line' else '订单级 ETA（仅干预订单承担流失）'
    print(f"\n==== {label} ====")
    print(f"  - ETA 生效口径: {scope_cn}")
    print(f"  - 计入成本的订单总量: {total_target} 单")
    print(f"  - 平均延长天数: {df_calc['实际延长天数'].mean():.2f} 天")
    print(f"  - 整体挽救率(对计入成本的订单): {real_rescue_rate * 100:.2f}%")
    print(f"  - 保护的 GMV: {protected_gmv:.2f} 元")
    print(f"  - 预期 GMV 损失: {avg_loss:.2f} 元")
    print(f"  - 95% 置信区间: [{np.percentile(gmv_loss_array, 2.5):.2f}, {np.percentile(gmv_loss_array, 97.5):.2f}]")
    print(f"  - 点估计净收益(保护-流失): {protected_gmv - base_loss_gmv:+.2f} 元")

    return gmv_loss_array, avg_loss, protected_gmv, base_loss_gmv


# ================= 提取样本并对比三套方案 =================
logistics_orders = df[df['超时因素'] == '运送延迟导致延期'].copy()

# 方案 A：全量订单统一 +2 天（线路级 ETA，成本由 72,907 单全部承担）
res_A_arr, loss_A, prot_A, base_A = run_monte_carlo(
    df, mode='fixed', fixed_days=2, scope='line', label="方案A：全量统一延时 2 天"
)

# 方案 B：仅纯物流延迟订单 +2 天（订单级 ETA，成本仅由 5,021 单承担）
res_B_arr, loss_B, prot_B, base_B = run_monte_carlo(
    logistics_orders, mode='fixed', fixed_days=2, scope='order', label="方案B：仅纯物流延迟订单延时 2 天"
)

# 方案 C：净收益最优动态阶梯（线路级 ETA，成本由各线路全部订单承担）
res_C_arr, loss_C, prot_C, base_C = run_monte_carlo(
    df, mode='dynamic', scope='line', label="方案C：净收益最优动态阶梯延时"
)

# ================= 结果汇总 =================
print("\n" + "=" * 46)
print("📊 【核心业务账本】（点估计，保护 GMV - 流失 GMV）")
for name, p, l in [("方案A 全量统一+2天", prot_A, base_A),
                   ("方案B 仅物流单+2天", prot_B, base_B),
                   ("方案C 净收益最优阶梯", prot_C, base_C)]:
    print(f"  {name}: 保护 {p:>10.2f} 元 | 流失 {l:>10.2f} 元 | 净 {p - l:>+10.2f} 元")
print("-" * 46)
extra_protected = prot_C - prot_B
extra_loss = base_C - base_B
net_benefit = extra_protected - extra_loss
print(f"✅ 方案C 相对方案B 额外保护 GMV: {extra_protected:+.2f} 元")
print(f"❌ 方案C 相对方案B 额外流失 GMV: {extra_loss:+.2f} 元")
print(f"💰 方案C 相对方案B 净差额: {net_benefit:+.2f} 元")
if net_benefit > 0:
    print("🎉 结论：在各自 ETA 生效口径下，方案C 的净收益优于方案B。")
else:
    print("⚠️ 结论：方案C 的转化代价高于其额外挽回的体验价值，建议以方案B 为准。")
print("=" * 46 + "\n")

# ================= 可视化 =================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, arr, title, color in [(axes[0], res_A_arr, '方案A：全量统一 +2 天', 'salmon'),
                              (axes[1], res_B_arr, '方案B：仅物流延迟单 +2 天', 'orange'),
                              (axes[2], res_C_arr, '方案C：净收益最优动态阶梯', 'lightgreen')]:
    ax.hist(arr, bins=50, color=color, edgecolor='black', alpha=0.7)
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
               '延迟时长_P50', '延迟时长_P75', '延迟时长_P90',
               '推荐延迟天数', '预期挽救单量', '预期保护GMV', '预期流失GMV',
               '预期净收益GMV', '预期转化流失率', '是否建议干预']
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
