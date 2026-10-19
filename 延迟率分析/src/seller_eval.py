import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ========== 1.读取数据 ==========
df = pd.read_csv(BASE_DIR / 'processed_data' / "延迟率与商家发货时间关系.csv", encoding="utf-8-sig")

# 预处理：签收超时订单=0时，收货超时订单中发货超时占比填充0，避免分母为0
df.loc[df["超时订单数"] == 0, "收货超时订单中发货超时占比"] = 0

# 过滤：总订单≥5，剔除小样本极端噪声
df = df[df["总订单"] >= 5].copy()
print(f"过滤小样本后卖家样本量：{len(df)}")

# ========== 2.业务阈值（可根据数据分布自行调整） ==========
T1 = 10    # 签收超时率阈值：>10%认为签收超时偏高
T2 = 50   # 签收超时订单内，发货超时占比阈值：>60%代表超时主因是发货
T3 = 15    # 发货超时率阈值：>15%代表卖家发货经常超时

def classify_seller(row):
    final_delay_rate = row["超时率"]
    cause_ratio = row["收货超时订单中发货超时占比"]
    ship_delay_rate = row["发货超时率"]

    if final_delay_rate > T1 and cause_ratio > T2:
        return "A_卖家主体严重风险_建议下架评估", "卖家发货"
    elif final_delay_rate > T1 and cause_ratio <= T2:
        return "B_送达超时非卖家责任_优化预测/快递", "快递/预估算法"
    elif final_delay_rate <= T1 and ship_delay_rate > T3:
        return "C_发货行为违规预警_约谈警告", "卖家发货（潜在风险）"
    else:
        return "D_优质卖家_无需干预", "无明显风险"

# 应用规则
res = df.apply(lambda x: pd.Series(classify_seller(x)), axis=1)
df["风险等级"] = res[0]
df["超时归因"] = res[1]

# ========== 统计各类卖家数量 ==========
stat = df["风险等级"].value_counts()
print("\n====卖家风险分类统计=====")
print(stat)

# ========== 可视化各类卖家分布散点图 ==========
plt.figure(figsize=(8,6))
color_map = {
    "A_卖家主体严重风险_建议下架评估":"red",
    "B_送达超时非卖家责任_优化预测/快递":"orange",
    "C_发货行为违规预警_约谈警告":"gold",
    "D_优质卖家_无需干预":"green"
}
label_set = set()
for risk_name, color in color_map.items():
    sub = df[df["风险等级"] == risk_name]
    if len(sub)>0:
        plt.scatter(sub["超时率"], sub["发货超时率"], c=color, label=risk_name, alpha=0.6)

plt.xlabel("签收超时率")
plt.ylabel("发货超时率")
plt.title("卖家履约风险分布（业务规则归因）")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.show()

# ========== 按风险分组，输出4个csv文件到output ==========
output_dir = BASE_DIR / "商家层面分析结果"
output_dir.mkdir(exist_ok=True)

group_list = df["风险等级"].unique()
total_rows = 0
print("\n====分组文件输出清单=====")
for group_name in group_list:
    sub_df = df[df["风险等级"] == group_name]
    safe_filename = group_name.replace("/","_")
    save_path = output_dir / f"{safe_filename}.csv"
    sub_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    total_rows += len(sub_df)
    print(f"{group_name} -> {save_path.name}  | 卖家数量：{len(sub_df)}")

print(f"\n总行数校验：拆分{total_rows}，原始{len(df)}，校验{'✅' if total_rows == len(df) else '❌'}")

# 输出完整总表
df.to_csv(output_dir / "seller_risk_rule_full.csv", index=False, encoding="utf-8-sig")
print(f"\n完整风险结果保存到：{output_dir / 'seller_risk_rule_full.csv'}")
