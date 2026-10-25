from pathlib import Path

# ============================ 1. 路径 ============================
ANALYSIS_DIR = Path(__file__).resolve().parent          # .../延迟率分析/GMV_analysis
ROOT_DIR = ANALYSIS_DIR.parent                          # .../延迟率分析
ETA_DIR = ROOT_DIR.parent / "运送预期算法优化"           # .../kaggle巴西电商分析/运送预期算法优化

# 输入
GMV_CSV = ROOT_DIR / "processed_data" / "GMV分析.csv"
ROUTE_CSV = ROOT_DIR / "processed_data" / "各物流线路延迟率分析.csv"
ETA_CSV = ETA_DIR / "output" / "LightGBM预测数据.csv"
OOC_CSV = ETA_DIR / "output" / "ooc_predictions.csv"

# 输出（本模块独立存放，不污染既有目录）
OUTPUT_DIR = ANALYSIS_DIR / "output"
ASSET_DIR = ANALYSIS_DIR / "assets"

LEDGER_CSV_NAME = "gmv_scenario_ledger.csv"             # 主账本（点估计）
MONTE_CARLO_CSV_NAME = "gmv_scenario_monte_carlo.csv"   # 蒙特卡洛结果（含 95% 区间）
ROBUSTNESS_CSV_NAME = "gmv_scope_robustness.csv"        # 三口径稳健性
SENSITIVITY_CSV_NAME = "gmv_sensitivity.csv"            # 参数敏感性
ORDER_DETAIL_CSV_NAME = "gmv_scenario_order_detail.csv" # 逐单明细（留档，可按需关闭）

# ============================ 2. 字段命名 ============================
# GMV 侧字段（延迟率分析项目）
COL = {
    "order_time": "下单时间",          # GMV 表里只到日，精确时刻取自 ETA 表
    "price": "总价格",                 # 订单 GMV（价格+运费）
    "promise": "预计总运送周期",        # 原算法承诺总周期（天，julianday 差值）
    "actual": "实际总运送周期",         # 实际总周期（天）
    "delay_days": "延迟天数_天",        # 相对原算法的超时天数（整数天）
    "status": "送达状态",              # 提前 / 准时 / 延迟
    "cause": "超时因素",               # 未超时 / 运送延迟导致延期 / 发货延迟…
    "score": "订单评分",
    "seller_state": "发货州",
    "customer_state": "收货州",
}
STATUS_DELAYED = "延迟"
CAUSE_LOGISTICS = "运送延迟导致延期"   # 物流与算法责任，是延长 ETA 唯一能救的部分

# ETA 预测表字段（运送预期算法优化项目）
COL_ETA = {
    "order_time": "下单时间",          # 精确到秒，用于时间切分
    "promise": "预期总周期",            # 即 GMV 表的 预计总运送周期
    "actual": "总运送周期",
    "score": "订单评分",
    "seller_state": "发货州",
    "customer_state": "收货州",
}
MODEL_ETA_SUFFIX = "预期总周期"        # 例：PseudoHuber_Under20x预期总周期
OOC_ETA_SUFFIX = "_ooc_预期总周期"     # ooc_predictions.csv 里的列名后缀

# ============================ 3. 对齐校验 ============================
# GMV 表与 ETA 表同源且同序，但本模块不做假设，逐行校验后才会往下走
ALIGN_CHECK_NUMERIC = [                # (GMV列, ETA列) 容差 1e-6
    ("预计总运送周期", "预期总周期"),
    ("实际总运送周期", "总运送周期"),
    ("两地直线距离", "两地直线距离"),
    ("价格", "价格"),
    ("运费", "运费"),
]
ALIGN_CHECK_OBJECT = [                 # 必须逐行完全相等
    "seller_id", "customer_unique_id", "发货州", "收货州",
]

# ============================ 4. 新 ETA 模型清单 ============================
# 与运送预期算法优化/src/config.py 的 under_weights 保持一致
MODEL_SOURCE_COLS = [
    "L1",
    "PseudoHuber_Under5x",
    "PseudoHuber_Under20x",
    "PseudoHuber_Under40x",
]
SELECTED_MODEL = "PseudoHuber_Under20x"    # 当前选定要替换原算法的模型
MODEL_DISPLAY_NAME = {                     # 图表上的中文名，避免满屏英文
    "L1": "新模型 L1",
    "PseudoHuber_Under5x": "新模型 5x",
    "PseudoHuber_Under20x": "新模型 20x（选定）",
    "PseudoHuber_Under40x": "新模型 40x",
}

# 折外预测统一由 运送预期算法优化/src/ooc_val.py 产出：它已按 row_pos 记录行位置、
# 并完成了向上取整到 00:00，本模块直接按行号挂回，不重复做行序与取整处理。

# ============================ 5. 转化流失模型 ============================
# 参数沿用 precise_gmv_simulated.py，全部为外部文献值，待平台真实 A/B 实验替换
DAILY_LOSS_RATE = 0.018      # 每多承诺 1 天的转化率流失基准
DAILY_LOSS_STD = 0.005       # 该基准本身的不确定性，蒙特卡洛中作为抽样标准差
VOLUME_SWING = 0.10          # 单量波动 ±10%（有放回抽样）

N_SIMULATIONS = 10000        # 蒙特卡洛次数
RANDOM_SEED = 42

# 【新增的两个保守化设置】——换算法与加缓冲不同，必须单独处理
# 1) 等效天数封顶：新模型会把个别订单的承诺从 40 天砍到 10 天，原始复利模型
#    外推到等效 ±几十天时已经不可信（会算出「单单转化 +200%」这种数）。
#    超过 MAX_EFF_DAYS 的部分按封顶计，防止极端订单主导全盘。
MAX_EFF_DAYS = 14.0
# 2) 缩短收益折扣：文献值只校准了「加时→流失」的方向，「减时→涨转化」是否对称并无依据。
#    SHORTEN_GAIN_RATIO 用于给缩短承诺带来的转化收益打折，1.0=对称（乐观），0.5=只认一半。
SHORTEN_GAIN_RATIO = 1.0

# 敏感性网格（主账本用上面的默认值，这里给出替代方案）
# 注意 None 表示「取上面的 MAX_EFF_DAYS」，若要真正不封顶请传 float("inf")。
SHORTEN_GAIN_RATIO_GRID = [1.0, 0.6, 0.3]
EFF_CAP_GRID = [14.0, 7.0, float("inf")]

# ============================ 6. 评估口径（稳健性三口径） ============================
# 新模型的预测值里，训练集区间的行是「样本内」的，延迟率会被系统性低估，
# 因此必须同时给出三个口径，避免被乐观数字误导：
#   · 全量         —— 与部署口径一致（模型用全量数据训练后再回填），但含样本内乐观
#   · 时间外测试期 —— order_time 后 20%，模型只见过之前的数据，与 lgb_model.py 测试集同源
#   · OOF 折外     —— 5 折前向链式 CV，最接近「持续增量训练 + 预测新单」的真实部署
SCOPES = ["全量", "时间外测试期", "OOF折外"]
PRIMARY_SCOPE = "全量"          # 主账本口径
TEST_SIZE = 0.20                # 与运送预期算法优化/src/config.py 的 test_size 保持一致
ROBUSTNESS_MODELS = MODEL_SOURCE_COLS   # 稳健性表里要跑的模型

# ============================ 7. 场景清单 ============================
# 每个场景 = 一套「客户看到的承诺时长」= ETA来源 + 延时规则。字段说明：
#   source : 原算法 / 模型列名
#   extend : none（不加）/ fixed:N（全量加 N 天）/ fixed:N+logistics（只给物流延迟单加，需上帝视角）
#            / route（按各物流线路延迟率分析.csv 的推荐延迟天数）
# 注：转化损益只在「承诺真的变了」的订单上计，因此转化侧无需额外指定成本承担集合，
#     承诺没变的订单 Δ=0，贡献天然为 0。
SCENARIOS = [
    {"name": "① 原算法（基准）", "source": "原算法", "extend": "none"},

    {"name": "② 延时·方案A 全量+2天", "source": "原算法", "extend": "fixed", "days": 2},
    {"name": "③ 延时·方案B 仅物流延迟单+2天", "source": "原算法", "extend": "fixed", "days": 2,
     "target": "logistics"},
    {"name": "④ 延时·方案C 线路阶梯", "source": "原算法", "extend": "route"},

    {"name": "⑤ 替换·新模型 L1", "source": "L1", "extend": "none"},
    {"name": "⑥ 替换·新模型 5x", "source": "PseudoHuber_Under5x", "extend": "none"},
    {"name": "⑦ 替换·新模型 20x（选定）", "source": "PseudoHuber_Under20x", "extend": "none"},
    {"name": "⑧ 替换·新模型 40x", "source": "PseudoHuber_Under40x", "extend": "none"},

    {"name": "⑨ 组合·新模型 20x + 线路阶梯", "source": "PseudoHuber_Under20x", "extend": "route"},
    {"name": "⑩ 组合·新模型 20x + 全量2天", "source": "PseudoHuber_Under20x", "extend": "fixed", "days": 2},
]

# ============================ 8. 图表 ============================
FONT_SANS = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]   # 与延迟率分析既有脚本一致
FIGURE_SIZE = (13, 6)
FIGURE_SIZE_TRIPLE = (18, 5)
DPI = 150
PLOT_BINS = 50
EXPORT_ORDER_DETAIL = True      # 是否导出逐单明细（72,907 行 × 多列，约 10MB）
