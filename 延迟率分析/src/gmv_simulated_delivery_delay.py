from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据
BASE_DIR = Path(__file__).parent.parent
df = pd.read_csv(BASE_DIR / 'processed_data' / "GMV分析.csv", encoding="utf-8-sig")

# 2. 封装通用的蒙特卡洛模拟函数
def run_monte_carlo_simulation(target_df, extension_days, label):
    total_target = len(target_df)
    if total_target == 0:
        return None, 0, 0, 0

    # 【关键修正】：只有原本“送达状态为延迟”且“延迟天数<=延长天数”的订单，才算真正被挽救
    rescued_mask = (target_df['送达状态'] == '延迟') & (target_df['延迟天数_天'] <= extension_days)
    real_rescue_rate = rescued_mask.sum() / total_target
    protected_gmv_value = target_df.loc[rescued_mask, '总价格'].sum()

    N_SIMULATIONS = 10000
    gmv_loss_list = []

    mean_loss_rate = 0.018 * extension_days
    std_loss_rate = 0.005 * np.sqrt(extension_days)

    for i in range(N_SIMULATIONS):
        # 订单量波动 ±10%
        simulated_count = int(total_target * np.random.uniform(0.9, 1.1))
        sampled_orders = target_df.sample(n=simulated_count, replace=True)
        sample_total_gmv = sampled_orders['总价格'].sum()

        # 转化率流失波动
        simulated_loss_rate = max(0, np.random.normal(mean_loss_rate, std_loss_rate))
        gmv_loss_list.append(sample_total_gmv * simulated_loss_rate)

    gmv_loss_array = np.array(gmv_loss_list)

    print(f"\n==== {label} 模拟结果 ====")
    print(f"历史目标订单总数: {total_target}")
    print(f"【数据提取】延长 {extension_days} 天 ETA：")
    print(f"  - 整体挽救率 (救回单/干预单): {real_rescue_rate * 100:.2f}%")
    print(f"  - 被挽救的 GMV 保护量: {protected_gmv_value:.2f} 元")
    print(f"【模拟推演】{N_SIMULATIONS} 次模拟：")
    print(f"  - 平均预期 GMV 损失: {np.mean(gmv_loss_array):.2f} 元")
    print(f"  - 95% 置信区间: [{np.percentile(gmv_loss_array, 2.5):.2f}, {np.percentile(gmv_loss_array, 97.5):.2f}]")

    return gmv_loss_array, real_rescue_rate, np.mean(gmv_loss_array), protected_gmv_value

# ================= 3. 提取两套不同的样本 =================
extension_days = 2

# 方案 A：全量所有订单（一刀切）
all_orders = df.copy()

# 方案 B：仅限“运送延迟导致延期”（纯物流责任，精准干预）
delayed_logistics = df[df['超时因素'] == '运送延迟导致延期'].copy()

# ================= 4. 分别跑模拟 =================
res_A = run_monte_carlo_simulation(all_orders, extension_days, "方案A：全量所有订单一刀切 (最粗暴)")
res_B = run_monte_carlo_simulation(delayed_logistics, extension_days, "方案B：仅干预纯物流延迟订单 (对症下药)")

# ================= 5. 双图对比可视化 =================
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 图 A
axes[0].hist(res_A[0], bins=50, color='salmon', edgecolor='black', alpha=0.7)
axes[0].axvline(res_A[2], color='red', linestyle='dashed', linewidth=2, label=f'平均损失: {res_A[2]:.2f}')
axes[0].set_title('方案A：全量订单一刀切 (转化率大出血)')
axes[0].set_xlabel('GMV 损失金额')
axes[0].set_ylabel('模拟频次')
axes[0].legend()

# 图 B
axes[1].hist(res_B[0], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
axes[1].axvline(res_B[2], color='darkgreen', linestyle='dashed', linewidth=2, label=f'平均损失: {res_B[2]:.2f}')
axes[1].set_title('方案B：纯物流延迟订单精准干预')
axes[1].set_xlabel('GMV 损失金额')
axes[1].set_ylabel('模拟频次')
axes[1].legend()

plt.tight_layout()
plt.show()