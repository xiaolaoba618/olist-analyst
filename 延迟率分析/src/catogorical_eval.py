import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ========== 1.读取数据 ==========
# 读取上一轮导出的商品类型聚合数据
df = pd.read_csv(BASE_DIR / 'processed_data' / "延迟率与商品类型关系.csv", encoding="utf-8-sig")

# 【关键预处理】：将字符串类型的比例和评分转回浮点数以便计算
# 假设之前的列名为：'该品类超时率', '该品类平均评分', '该品类总订单数'
df['该品类超时率'] = df['该品类超时率'].astype(float)
df['该品类平均评分'] = df['该品类平均评分'].astype(float)

# 过滤：总订单≥5，剔除长尾冷门商品噪声
df = df[df["该品类总订单数"] >= 5].copy()
print(f"过滤小样本后商品品类样本量：{len(df)}")


# ========== 2.业务阈值 ==========
T_RATING_GOOD = 4.2  # 优质评分标准
T_DELAY_GOOD = 8     # 优质超时率标准 (%) -> 小于 8%
T_RATING_BAD = 4.0   # 受严重影响评分标准
T_DELAY_BAD = 8.9    # 高延迟超时率标准 (%) -> 大于 8.9%

def classify_category(row):
    rating = row["该品类平均评分"]
    delay_rate = row["该品类超时率"]

    # A类：优质商品
    if rating > T_RATING_GOOD and delay_rate < T_DELAY_GOOD:
        return "A_优质商品_不易超时且评分高"
    
    # B类：评分受超时率影响最大（评分低且超时率高）
    elif rating < T_RATING_BAD and delay_rate > T_DELAY_BAD:
        return "B_评分受超时率影响最大"
    
    # C类：警惕部分（超时率高，但评分暂时还撑在 4.0 或以上，未完全崩塌）
    elif delay_rate > T_DELAY_BAD and rating >= T_RATING_BAD:
        return "C_警惕商品_高延迟但暂未严重影响评分"
    
    # D类：普通商品（不符合上述极端的其他情况）
    else:
        return "D_普通商品_表现正常"

# 应用规则
df["风险等级"] = df.apply(classify_category, axis=1)

# ========== 3.统计各类商品数量 ==========
stat = df["风险等级"].value_counts()
print("\n====商品分类统计=====")
print(stat)


# ========== 4.可视化各类商品分布散点图 ==========
plt.figure(figsize=(10, 6))
color_map = {
    "A_优质商品_不易超时且评分高": "green",
    "B_评分受超时率影响最大": "red",
    "C_警惕商品_高延迟但暂未严重影响评分": "gold",
    "D_普通商品_表现正常": "gray"
}

for risk_name, color in color_map.items():
    sub = df[df["风险等级"] == risk_name]
    if len(sub) > 0:
        plt.scatter(sub["该品类超时率"], sub["该品类平均评分"], 
                    c=color, label=risk_name, alpha=0.7, edgecolors='w', s=60)

# 画辅助阈值线，帮助直观看到分类边界
plt.axhline(y=T_RATING_GOOD, color='g', linestyle='--', alpha=0.3)
plt.axhline(y=T_RATING_BAD, color='r', linestyle='--', alpha=0.3)
plt.axvline(x=T_DELAY_GOOD, color='g', linestyle='--', alpha=0.3)
plt.axvline(x=T_DELAY_BAD, color='r', linestyle='--', alpha=0.3)

plt.xlabel("该品类超时率 (%)")
plt.ylabel("该品类平均评分")
plt.title("商品品类：超时率与评分关系分布（业务规则归因）")
plt.legend(fontsize=9, loc='lower left')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


# ========== 5.按风险分组，输出csv文件 ==========
output_dir = BASE_DIR / "商品层面分析结果"
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
    print(f"{group_name} -> {save_path.name}  | 商品品类数量：{len(sub_df)}")

print(f"\n总行数校验：拆分{total_rows}，原始{len(df)}，校验{'✅' if total_rows == len(df) else '❌'}")

# 输出完整总表
df.to_csv(output_dir / "category_risk_rule_full.csv", index=False, encoding="utf-8-sig")
print(f"\n完整风险结果保存到：{output_dir / 'category_risk_rule_full.csv'}")
