from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).parent.parent
df = pd.read_csv(BASE_DIR / 'processed_data' / "GMV分析.csv", encoding="utf-8-sig")
# 【修改】：读取 V3 数据
route_df = pd.read_csv(BASE_DIR / 'processed_data' / "各物流线路延迟率分析.csv", encoding="utf-8-sig")

route_map = route_df.set_index(['发货州', '收货州'])['推荐延迟天数'].to_dict()
df['推荐延长天数'] = df.apply(lambda x: route_map.get((x['发货州'], x['收货州']), 0), axis=1)

# ================= 3. 封装通用蒙特卡洛函数 =================
def run_monte_carlo(target_df, mode='fixed', fixed_days=2, label=""):
    total_target = len(target_df)
    if total_target == 0:
        return None, 0, 0

    if mode == 'fixed':
        target_df = target_df.copy()
        target_df['实际延长天数'] = fixed_days
    else:
        target_df = target_df.copy()
        target_df['实际延长天数'] = target_df['推荐延长天数']

    rescued_mask = (target_df['送达状态'] == '延迟') & (target_df['延迟天数_天'] <= target_df['实际延长天数'])
    protected_gmv_value = target_df.loc[rescued_mask, '总价格'].sum()
    real_rescue_rate = rescued_mask.sum() / total_target

    N_SIMULATIONS = 10000
    gmv_loss_list = []

    for _ in range(N_SIMULATIONS):
        simulated_count = int(total_target * np.random.uniform(0.9, 1.1))
        sampled_orders = target_df.sample(n=simulated_count, replace=True)
        sample_total_gmv = sampled_orders['总价格'].sum()

        # 转化率流失计算
        mean_loss_rate = 0.018 * sampled_orders['实际延长天数'].mean()
        std_loss_rate = 0.005 * np.sqrt(sampled_orders['实际延长天数'].mean())

        simulated_loss_rate = max(0, np.random.normal(mean_loss_rate, std_loss_rate))
        gmv_loss_list.append(sample_total_gmv * simulated_loss_rate)

    gmv_loss_array = np.array(gmv_loss_list)
    avg_loss = np.mean(gmv_loss_array)

    print(f"\n==== {label} ====")
    print(f"  - 干预订单总量: {total_target} 单")
    print(f"  - 平均延长天数: {target_df['实际延长天数'].mean():.2f} 天")
    print(f"  - 整体挽救率: {real_rescue_rate * 100:.2f}%")
    print(f"  - 保护的 GMV: {protected_gmv_value:.2f} 元")
    print(f"  - 预期 GMV 损失: {avg_loss:.2f} 元")
    print(f"  - 95% 置信区间: [{np.percentile(gmv_loss_array, 2.5):.2f}, {np.percentile(gmv_loss_array, 97.5):.2f}]")

    return gmv_loss_array, avg_loss, protected_gmv_value

# ================= 4. 提取样本并对比 =================
logistics_orders = df[df['超时因素'] == '运送延迟导致延期'].copy()

# 方案 A：统一延时 2 天
res_A_arr, loss_A, prot_A = run_monte_carlo(
    target_df=logistics_orders, mode='fixed', fixed_days=2, label="方案A：统一延时 2 天 (基准)"
)

# 方案 B：V3版动态阶梯延时
df_dynamic = logistics_orders[logistics_orders['推荐延长天数'] > 0].copy()
res_B_arr, loss_B, prot_B = run_monte_carlo(
    target_df=df_dynamic, mode='dynamic', label="方案B：基于 P75 的高挽救率动态延时"
)

# ================= 5. 结果汇总与可视化 =================
# ================= 5. 结果汇总与可视化 =================
# 1. 转化成本差额（代码原来的算法）
diff_cost = loss_A - loss_B

# 2. 综合净收益（你的手算算法）
extra_protected_gmv = prot_B - prot_A
extra_loss_cost = loss_B - loss_A
net_benefit = extra_protected_gmv - extra_loss_cost

print("\n" + "="*40)
print("📊 【核心业务账本】")
print(f"方案A (统一延2天): 保护 {prot_A:.2f} 元 | 流失 {loss_A:.2f} 元")
print(f"方案B (动态阶梯延时): 保护 {prot_B:.2f} 元 | 流失 {loss_B:.2f} 元")
print("-" * 40)
print(f"✅ 额外保护的 GMV: +{extra_protected_gmv:.2f} 元")
print(f"❌ 额外流失的 GMV: -{extra_loss_cost:.2f} 元")
print(f"💰 【综合商业净收益】: {net_benefit:+.2f} 元")
if net_benefit > 0:
    print("🎉 结论：方案B 虽然牺牲了微小的转化率，但大幅挽回了高危 GMV，综合净收益为正！")
else:
    print("⚠️ 结论：方案B 过于激进，付出的转化成本超过了挽回的 GMV，建议调低封顶天数。")
print("="*40 + "\n")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].hist(res_A_arr, bins=50, color='salmon', edgecolor='black', alpha=0.7)
axes[0].axvline(loss_A, color='red', linestyle='dashed', linewidth=2, label=f'平均损失: {loss_A:.2f}')
axes[0].set_title('方案A：统一延 2 天')
axes[0].legend()

axes[1].hist(res_B_arr, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
axes[1].axvline(loss_B, color='darkgreen', linestyle='dashed', linewidth=2, label=f'平均损失: {loss_B:.2f}')
axes[1].set_title('方案B：P75 高挽救率阶梯延时')
axes[1].legend()

plt.tight_layout()
plt.show()

# ================= 6. 导出各线路推荐延迟天数 CSV =================
# 直接利用当前的 route_df 复制一份用于导出
route_output = route_df.copy()

# 按照业务需求，调整列的顺序（把 P75 和推荐天数放进去）
columns_order = ['发货州', '收货州', '平均距离_km', '总订单数', '物流延迟率', '延迟时长_P75', '推荐延迟天数']
# 过滤掉当前数据里不存在的列，防止报错
existing_cols = [col for col in columns_order if col in route_output.columns]
route_output = route_output[existing_cols]

# 按“推荐延迟天数”降序、“物流延迟率”降序排列，把最恶劣的线路排在前面
route_output = route_output.sort_values(by=['推荐延迟天数', '物流延迟率'], ascending=[False, False])

# 创建输出目录并保存
output_dir = BASE_DIR / '线路延迟预送时间推荐结果'
output_dir.mkdir(parents=True, exist_ok=True)
route_csv_path = output_dir / '基于P75的高挽救率线路延迟配置表.csv'

route_output.to_csv(route_csv_path, index=False, encoding='utf-8-sig')

print(f"\n✅ 线路推荐延迟配置表已成功导出至：{route_csv_path}")
print("==== 延迟最严重 (需要干预最久) 的 Top 10 线路配置 ====")
print(route_output.head(10).to_string(index=False))