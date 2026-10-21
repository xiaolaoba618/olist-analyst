import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体，防止图表中文乱码 (Windows用SimHei，Mac用Arial Unicode MS)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).parent.parent
print(BASE_DIR)

df = pd.read_csv(BASE_DIR / 'data' / "订单时间段分析_用户画像清洗后.csv", encoding="utf-8-sig")


# 1. 时间格式转换
df['下单时间'] = pd.to_datetime(df['下单时间'])

# 2. 找出出现过两次及以上的 customer_unique_id
repeat_customers = df['customer_unique_id'].value_counts()
repeat_customers = repeat_customers[repeat_customers >= 2].index
df_repeat = df[df['customer_unique_id'].isin(repeat_customers)].copy()

# ================= 关键修改部分 =================
# 3. 按用户和时间排序
df_repeat = df_repeat.sort_values(by=['customer_unique_id', '下单时间'])

# 4. 生成组内序号 (0, 1, 2...)，代表是该用户的第几单
df_repeat['order_seq'] = df_repeat.groupby('customer_unique_id').cumcount()

# 5. 提取第一单 (seq=0) 和 第二单 (seq=1)，并直接重命名时间列
df_first = df_repeat[df_repeat['order_seq'] == 0][['customer_unique_id', '下单时间']].rename(columns={'下单时间': '首单时间'})
df_second = df_repeat[df_repeat['order_seq'] == 1][['customer_unique_id', '下单时间']].rename(columns={'下单时间': '二单时间'})

# 6. 合并两张表 (因为已经过滤了>=2次的用户，这里一定会一一对应)
df_interval = pd.merge(df_first, df_second, on='customer_unique_id', how='inner')

# 7. 计算间隔天数
df_interval['间隔天数'] = (df_interval['二单时间'] - df_interval['首单时间']).dt.days
# ================================================

# ================= 8. 筛选 1-39 天的数据并计算概率 =================
# 排除 0 天，且限制在 39 天以内
df_short_term = df_interval[(df_interval['间隔天数'] > 0) & (df_interval['间隔天数'] <= 39)].copy()
interval_counts_39d = df_short_term['间隔天数'].value_counts(normalize=True).sort_index()

# ================= 9. 开始可视化 =================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- 图1：概率折线图 (1-39天) ---
axes[0].plot(interval_counts_39d.index, interval_counts_39d.values,
             color='#1f77b4', linewidth=2) # 去除 marker 让线条更平滑，看清整体趋势
axes[0].fill_between(interval_counts_39d.index, interval_counts_39d.values, color='#1f77b4', alpha=0.15)

# 加上趋势线（多项式拟合）帮助看出整体下滑或周期规律
import numpy as np
z = np.polyfit(interval_counts_39d.index, interval_counts_39d.values, 3)
p = np.poly1d(z)
axes[0].plot(interval_counts_39d.index, p(interval_counts_39d.index), "r--", alpha=0.8, label="趋势线")

axes[0].set_title('复购间隔天数概率分布 (1-39天)')
axes[0].set_xlabel('间隔天数')
axes[0].set_ylabel('概率 (占比)')
# X 轴每 5 天显示一个刻度，避免拥挤
axes[0].set_xticks(range(1, 40, 5))
axes[0].grid(alpha=0.3)
axes[0].legend()

# --- 图2：概率密度直方图 (1-39天) ---
sns.histplot(data=df_short_term,
             x='间隔天数',
             stat='probability',
             bins=39, # 跨度变大，适当增加 bins
             kde=True,
             ax=axes[1],
             color='#2ca02c')
axes[1].set_title('复购间隔天数概率分布 (1-39天 密度)')
axes[1].set_xlabel('间隔天数')
axes[1].set_ylabel('概率')
axes[1].set_xticks(range(1, 40, 5))

plt.tight_layout()
plt.show()

# ================= 10. 输出核心业务指标 =================
print(f"参与计算的总复购用户数 (含0天): {len(df_interval)}")
print(f"排除0天，且集中在 1-39 天内复购的用户数: {len(df_short_term)}")
print(f"占总体复购用户比例: {len(df_short_term) / len(df_interval) * 100:.2f}%")

# 寻找局部峰值 (Peaks)
# 我们可以提取出概率最高的几天
top_5_days = interval_counts_39d.nlargest(5)
print("\n📊 1-39 天内概率最高的复购天数 Top 5:")
for days, prob in top_5_days.items():
    print(f" - 第 {days} 天: 占比 {prob * 100:.2f}%")

# 按周聚合看周期规律
df_short_term['复购周数'] = (df_short_term['间隔天数'] / 7).astype(int) + 1
weekly_counts = df_short_term['复购周数'].value_counts(normalize=True).sort_index()
print("\n📅 按周汇总的复购概率分布:")
for week, prob in weekly_counts.items():
    print(f" - 第 {week} 周: 占比 {prob * 100:.2f}%")

# ================= 11. 新增：1-120 天长期复购趋势图 =================

# 筛选 1-120 天的数据
df_long_term = df_interval[(df_interval['间隔天数'] > 0) & (df_interval['间隔天数'] <= 120)].copy()

# 计算 1-120 天每天的概率分布
interval_counts_120d = df_long_term['间隔天数'].value_counts(normalize=True).sort_index()

# 补全 1-120 天，确保没有缺失的天数（如果没有复购则记为 0）
interval_counts_120d = interval_counts_120d.reindex(range(1, 121), fill_value=0)

# 计算 7 日滑动平均，用于平滑曲线看趋势
# center=True 表示取当前天前后各3天的平均值
smooth_counts = interval_counts_120d.rolling(window=7, center=True).mean()

# 开始绘图
plt.figure(figsize=(15, 6))

# 1. 画每日的原始概率（用浅色散点或细线，展示真实波动）
plt.plot(interval_counts_120d.index, interval_counts_120d.values,
         color='lightgray', linewidth=1, label='每日真实概率')

# 2. 画 7 日滑动平均曲线（用深色粗线，展示核心趋势）
plt.plot(smooth_counts.index, smooth_counts.values,
         color='#d62728', linewidth=2.5, label='7日滑动平均趋势')

# 3. 标注可能的长期周期节点（例如 30天、60天、90天）
for day in [30, 60, 90, 120]:
    plt.axvline(x=day, color='gray', linestyle='--', alpha=0.5)
    plt.text(day, max(interval_counts_120d.values)*0.9, f'{day}天',
             color='gray', ha='center', va='bottom', fontsize=10)

# 图表装饰
plt.title('复购间隔天数概率分布长期趋势 (1-120天)', fontsize=16)
plt.xlabel('间隔天数', fontsize=12)
plt.ylabel('复购概率 (占比)', fontsize=12)
plt.xlim(1, 120)
plt.xticks(range(0, 121, 10)) # X 轴每 10 天一个刻度
plt.grid(alpha=0.3, axis='y')
plt.legend(fontsize=11)

plt.tight_layout()
plt.show()

# ================= 12. 输出长期周期洞察 =================
# 计算前 30天、31-60天、61-90天、91-120天 的累计复购占比
def calc_period_prob(start, end):
    return interval_counts_120d.loc[start:end].sum() * 100

print("\n📈 长期复购周期分布概览 (占总复购人数比例):")
print(f" - 第 1-30 天: {calc_period_prob(1, 30):.2f}%")
print(f" - 第 31-60 天: {calc_period_prob(31, 60):.2f}%")
print(f" - 第 61-90 天: {calc_period_prob(61, 90):.2f}%")
print(f" - 第 91-120 天: {calc_period_prob(91, 120):.2f}%")

# 找出 1-120 天内，滑动平均曲线的 Top 3 峰值所在的大致区间
# 注意：找 peak 时排除掉最开头的几天（1-3天拆单噪音）
smooth_for_peak = smooth_counts.dropna().iloc[3:]
top_peaks = smooth_for_peak.nlargest(3)
print("\n🎯 长期趋势中的高峰区间 (7日平滑后概率最高):")
for day, prob in top_peaks.items():
    print(f" - 第 {int(day)} 天附近: 平均概率 {prob * 100:.3f}%")