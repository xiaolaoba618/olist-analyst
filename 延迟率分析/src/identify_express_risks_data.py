from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据
BASE_DIR = Path(__file__).parent.parent
df = pd.read_csv(BASE_DIR / 'processed_data' / "分级后总数据.csv", encoding="utf-8-sig")


# ================= 1. 计算每条线路的“大盘基础数据”（总单量、平均距离） =================
# 此时不筛超时原因，统计所有订单
route_base = df.groupby(['发货州', '收货州']).agg(
    总订单数=('发货州', 'size'),
    平均距离_km=('两地直线距离', 'mean')
).reset_index()

# ================= 2. 筛选“物流/算法”导致的超时订单 =================
# 注意：这里你需要确认你的超时原因列里到底叫什么
# 如果是 '物流/算法' 就直接写，如果是 '运送延迟导致延期' 就改一下
logistics_delays = df[df['超时原因'].isin(['物流/算法', '运送延迟导致延期'])].copy()

# ================= 3. 去尾处理（去除极端延迟天数） =================
# 直接算均值会被极端值（如延迟了100天）拉高。这里我们只保留延迟天数在 99 分位数以内的数据
threshold = logistics_delays['实际最终延迟时间'].quantile(0.99)
logistics_delays_trimmed = logistics_delays[logistics_delays['实际最终延迟时间'] <= threshold].copy()

print(f"极端延迟去尾阈值: {threshold:.2f} 天")

# ================= 4. 计算每条线路的“物流超时数据” =================
route_logistics = logistics_delays_trimmed.groupby(['发货州', '收货州']).agg(
    物流延迟单量=('发货州', 'size'),
    平均延迟时长_天=('实际最终延迟时间', 'mean')
).reset_index()

# ================= 5. 合并数据并计算最终指标 =================
route_analysis = pd.merge(route_base, route_logistics, on=['发货州', '收货州'], how='left')

# 填充没有物流延迟的线路为 0
route_analysis['物流延迟单量'] = route_analysis['物流延迟单量'].fillna(0)
route_analysis['平均延迟时长_天'] = route_analysis['平均延迟时长_天'].fillna(0)

# 计算物流延迟率 = 物流延迟单量 / 该线路总单量
route_analysis['物流延迟率'] = route_analysis['物流延迟单量'] / route_analysis['总订单数']

# ================= 6. 过滤小样本噪声 & 排序 =================
# 如果一条线路总共才 3 单，延误 1 单，延迟率就是 33%，这毫无意义。
# 我们只保留总订单数 >= 20 单的成熟线路进行分析
route_analysis_filtered = route_analysis[route_analysis['总订单数'] >= 20].copy()

# 按物流延迟率降序排列
route_analysis_filtered = route_analysis_filtered.sort_values(by='物流延迟率', ascending=False).reset_index(drop=True)

# 格式化输出百分比和保留小数
route_analysis_filtered['物流延迟率'] = (route_analysis_filtered['物流延迟率'] * 100).round(2).astype(str) + '%'
route_analysis_filtered['平均距离_km'] = route_analysis_filtered['平均距离_km'].round(2)
route_analysis_filtered['平均延迟时长_天'] = route_analysis_filtered['平均延迟时长_天'].round(2)

# ================= 7. 导出结果 =================
output_dir = BASE_DIR /'processed_data'
output_dir.mkdir(parents=True, exist_ok=True)
save_path = output_dir / '各物流线路延迟率分析.csv'
route_analysis_filtered.to_csv(save_path, index=False, encoding='utf-8-sig')

print(f"\n✅ 线路物流延迟分析已保存至：{save_path}")
print("\n==== 物流延迟率最高的 Top 10 线路 ====")
print(route_analysis_filtered.head(10))