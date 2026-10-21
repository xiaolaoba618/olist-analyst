import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ========== 1. 读取新生成的订单级归因数据 ==========
df = pd.read_csv(BASE_DIR / 'processed_data' / "商家延迟率与订单级归因表.csv", encoding="utf-8-sig")

# 【关键预处理】：将字符串转为浮点数
df['超时率'] = df['超时率'].astype(float)
df['发货超时率'] = df['发货超时率'].astype(float)
df['超时订单中发货因素占比'] = df['超时订单中发货因素占比'].astype(float)
df['潜在风险率'] = df['潜在风险率'].astype(float)  # 新增

# 预防性处理
df.loc[df["超时订单数(终态)"] == 0, "超时订单中发货因素占比"] = 0

# 过滤：总订单 ≥ 5
df = df[df["总订单"] >= 5].copy()
print(f"过滤小样本后，待分类卖家样本量：{len(df)}")

# ========== 2. 业务阈值 ==========
T1 = 10  # 签收超时率阈值
T2 = 60  # 发货因素占比阈值
T3 = 15  # 发货超时率阈值
T4 = 10  # 新增：潜在风险率阈值（超过10%的订单发货拖延但未导致超时，即预警）



def classify_seller_order_level(row):
    final_delay_rate = row["超时率"]
    cause_ratio = row["超时订单中发货因素占比"]
    ship_delay_rate = row["发货超时率"]
    potential_risk_rate = row["潜在风险率"]  # 新增：发货拖延但被快递救回的比例

    # A类：最终超时率高，且主要是发货导致的
    if final_delay_rate > T1 and cause_ratio > T2:
        return "A_订单级验证严重风险_建议下架评估", "卖家发货（订单级实锤）"

    # B类：最终超时率高，但主要是物流导致的
    elif final_delay_rate > T1 and cause_ratio <= T2:
        return "B_订单级验证非卖家责任_优化预测/快递", "快递/预估算法（订单级验证）"

    # C类：最终超时率低（或未超标），但发货行为极其不规范（包含两种违规情况）
    # 情况1：发货超时率 > 15% （发货经常导致延期）
    # 情况2：潜在风险率 > 10% （发货经常拖延，只是被快递救了回来）
    elif final_delay_rate <= T1 and (ship_delay_rate > T3 or potential_risk_rate > T4):
        return "C_订单级潜在发货违规_约谈警告", "卖家发货（潜在风险）"

    # D类：真正的优质卖家
    else:
        return "D_订单级优质卖家_无需干预", "无明显风险"


# 应用规则
res = df.apply(lambda x: pd.Series(classify_seller_order_level(x)), axis=1)
df["风险等级"] = res[0]
df["超时归因"] = res[1]

# ========== 3. 统计各类卖家数量 ==========
stat = df["风险等级"].value_counts()
print("\n====卖家订单级风险分类统计=====")
print(stat)

# ========== 4. 可视化各类卖家分布散点图 ==========
plt.figure(figsize=(8, 6))
color_map = {
    "A_订单级验证严重风险_建议下架评估": "red",
    "B_订单级验证非卖家责任_优化预测/快递": "orange",
    "C_订单级潜在发货违规_约谈警告": "gold",
    "D_订单级优质卖家_无需干预": "green"
}

for risk_name, color in color_map.items():
    sub = df[df["风险等级"] == risk_name]
    if len(sub) > 0:
        # 散点图的X轴可以用 潜在风险率，以更直观地展示 C 类的分布
        # 但为了与之前保持一致，还是用超时率和发货超时率
        plt.scatter(sub["超时率"], sub["发货超时率"], c=color, label=risk_name, alpha=0.6)

plt.xlabel("最终签收超时率 (%)")
plt.ylabel("发货超时率 (%)")
plt.title("卖家履约风险分布（订单级微观归因 & 包含潜在风险）")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.show()

# ========== 5. 按风险分组，输出文件 ==========
output_dir = BASE_DIR / "商家层面分析结果_订单级归因"
output_dir.mkdir(exist_ok=True)

group_list = df["风险等级"].unique()
total_rows = 0
print("\n====分组文件输出清单=====")
for group_name in group_list:
    sub_df = df[df["风险等级"] == group_name]
    safe_filename = group_name.replace("/", "_")
    save_path = output_dir / f"{safe_filename}.csv"
    sub_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    total_rows += len(sub_df)
    print(f"{group_name} -> {save_path.name}  | 卖家数量：{len(sub_df)}")

print(f"\n总行数校验：拆分{total_rows}，原始{len(df)}，校验{'✅' if total_rows == len(df) else '❌'}")

# 输出完整总表
full_save_path = output_dir / "seller_risk_rule_order_level_full.csv"
df.to_csv(full_save_path, index=False, encoding="utf-8-sig")
print(f"\n完整风险结果保存到：{full_save_path}")