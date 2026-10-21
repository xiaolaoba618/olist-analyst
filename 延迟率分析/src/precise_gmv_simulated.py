from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).parent.parent
df = pd.read_csv(BASE_DIR / 'processed_data' / "GMV分析.csv", encoding="utf-8-sig")
route_df = pd.read_csv(BASE_DIR / 'processed_data' / "各物流线路延迟率分析.csv", encoding="utf-8-sig")

# ================= 2. 数据准备 =================
# 直接读取第一块代码中算好的“推荐延迟天数”
route_map = route_df.set_index(['发货州', '收货州'])['推荐延迟天数'].to_dict()

df['推荐延长天数'] = df.apply(
    lambda x: route_map.get((x['发货州'], x['收货州']), 0), axis=1
)
df['推荐延长天数'] = df['推荐延长天数'].clip(lower=0)

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

# 方案 B：动态阶梯延时
df_dynamic = logistics_orders[logistics_orders['推荐延长天数'] > 0].copy()
res_B_arr, loss_B, prot_B = run_monte_carlo(
    target_df=df_dynamic, mode='dynamic', label="方案B：动态阶梯延时 (P90+封顶)"
)

# ================= 5. 结果汇总与可视化 =================
saved_cost = loss_A - loss_B
print(f"\n💰 【降本增效】：方案B 相比 方案A，额外节省了 {saved_cost:.2f} 元的 GMV 损失！")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].hist(res_A_arr, bins=50, color='salmon', edgecolor='black', alpha=0.7)
axes[0].axvline(loss_A, color='red', linestyle='dashed', linewidth=2, label=f'平均损失: {loss_A:.2f}')
axes[0].set_title('方案A：物流订单统一延 2 天 (简单粗暴)')
axes[0].legend()

axes[1].hist(res_B_arr, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
axes[1].axvline(loss_B, color='darkgreen', linestyle='dashed', linewidth=2, label=f'平均损失: {loss_B:.2f}')
axes[1].set_title('方案B：基于线路动态阶梯延时 (精准调优)')
axes[1].legend()

plt.tight_layout()
plt.show()

# ================= 6. 导出各线路推荐延迟天数配置表 =================
# 复制一份线路数据用于导出
route_output = route_df.copy()

# 按照推荐延迟天数降序，物流延迟率降序排列，把最恶劣、最需要紧急干预的线路排在最上面
# 注意：确保 route_df 中存在 '推荐延迟天数' 和 '物流延迟率' 这两个列
route_output = route_output.sort_values(
    by=['推荐延迟天数', '物流延迟率'],
    ascending=[False, False]
)

# 调整一下列的顺序，把业务核心指标放前面
columns_order = ['发货州', '收货州', '平均距离_km', '总订单数', '物流延迟率', '平均延迟时长_天', '推荐延迟天数']
# 确保这些列都存在，避免报错
existing_cols = [col for col in columns_order if col in route_output.columns]
route_output = route_output[existing_cols]

# 设置输出路径并保存
output_dir = BASE_DIR / '线路延迟预送时间推荐结果'
output_dir.mkdir(parents=True, exist_ok=True)
route_csv_path = output_dir / '各线路推荐延迟天数配置表.csv'

route_output.to_csv(route_csv_path, index=False, encoding='utf-8-sig')

print(f"\n✅ 线路推荐延迟配置表已成功导出至：{route_csv_path}")
print("==== 延迟最严重的 Top 10 线路配置 ====")
print(route_output.head(10).to_string(index=False))