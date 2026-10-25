#%% 1. 导入库
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_prepare import df as feature_data  # 复用已经构造好的特征表（英文列名，与 lgb_model.py 同源）
from config import CONFIG

#%% 2. 全局配置：与 lgb_model.py 保持一致


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

UNDER_WEIGHTS = CONFIG["under_weights"]
# 严重超时阈值（天）
SEVERE_UNDER_DAYS = CONFIG["severe_under_days"]
# 全量口径起点：2017 年 8 月数据只用于初期构造滑动窗口等特征工程，不进入模型与统计
FULL_START_TIME = pd.Timestamp(CONFIG["full_start_date"])

BASE_DIR = CONFIG["base_dir"]
OUTPUT_DIR, ASSET_DIR = Path(CONFIG["output_dir"]), Path(CONFIG["asset_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ASSET_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = CONFIG["model_dir"]
# LightGBM 无法直接打开中文路径下的模型文件，先复制到英文临时目录再加载（与 lgb_model.py 相同处理）
TEMP_MODEL_DIR = Path(r"G:\lgb_models_temp")
TEMP_MODEL_DIR.mkdir(parents=True, exist_ok=True)

#%% 3. 读取订单数据（全量原始数据，未经任何筛选）
df = pd.read_csv(BASE_DIR / "data" / "订单数据.csv", encoding="utf-8-sig")
df["下单时间"] = pd.to_datetime(df["下单时间"], format="mixed")

print(f"订单数据总量：{len(df)} | 特征表总量：{len(feature_data)}")
print("目标：把 lgb_model.py 中各模型的运送周期预测值回填到订单数据，并与原算法口径保持一致")

#%% 4. 行对齐：把特征表的每一行还原到订单数据的原始行位置
# delay_distribution.R 在最后一步删除了前 30 天的预热数据，并按 发货州、收货州、下单时间 排序；
# 该排序的前置次序为 arrange(seller_id, 下单时间)，再往后则沿用原始行序。
# 这里用同样的规则（稳定排序）复现特征表的行次序，从而把预测值准确挂回订单数据。
warm_mask = df["下单时间"] >= df["下单时间"].min() + pd.Timedelta(days=30)
aligned_index = (
    df.loc[warm_mask]
    .sort_values(["发货州", "收货州", "下单时间", "seller_id"], kind="stable")
    .index
)

# 校验：行数一致，且关键列逐行完全相同
if len(aligned_index) != len(feature_data):
    raise ValueError(
        f"行数对不上：订单数据筛出 {len(aligned_index)} 行，特征表有 {len(feature_data)} 行，"
        "请检查 delay_distribution.R 是否重新运行过。"
    )

check_pairs = [
    ("seller_id", "seller_id"), ("商品类型", "product_type"), ("付款方式", "payment_method"),
    ("发货州", "seller_state"), ("收货州", "customer_state"),
    ("总运送周期", "total_delivery_time"), ("预期总周期", "estimated_delivery_time"),
]
mismatch_count = 0
for raw_col, feat_col in check_pairs:
    left = pd.Series(df.loc[aligned_index, raw_col].to_numpy())
    right = pd.Series(feature_data[feat_col].to_numpy())
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        bad = ~np.isclose(left.astype(float), right.astype(float), atol=1e-6, equal_nan=True)
    else:
        bad = (left != right) & ~(left.isna() & right.isna())
    mismatch_count += int(np.asarray(bad).sum())

if mismatch_count > 0:
    raise ValueError(f"行对齐校验失败：共有 {mismatch_count} 处关键列不一致，请检查排序规则。")

print(f"行对齐校验通过：{len(aligned_index)} 行逐行一致（前 30 天预热的 "
      f"{int((~warm_mask).sum())} 行没有特征，导出时直接剔除）")

#%% 5. 复算 PseudoHuber 系列的中心化常数
# lgb_model.py 中 PseudoHuber 模型在「中心化后的对数目标」上训练，预测时需要把中心值加回来；
# 该常数未在模型文件中保存，因此按 lgb_model.py 完全相同的切分方式重算一次。
data = feature_data.copy()
data["order_time"] = pd.to_datetime(data["order_time"], errors="coerce")

required_cols = ["order_time", "total_delivery_time", "log_delivery_time", "estimated_delivery_time"]
missing_cols = [col for col in required_cols if col not in data.columns]
if missing_cols:
    raise ValueError(f"特征表缺少必要字段：{missing_cols}")

for col in ["total_delivery_time", "log_delivery_time", "estimated_delivery_time"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna(subset=required_cols).copy()
data = data[
    (data["total_delivery_time"] > 0)
    & (data["estimated_delivery_time"] > 0)
    & np.isfinite(data["log_delivery_time"])
    & np.isfinite(data["total_delivery_time"])
    & np.isfinite(data["estimated_delivery_time"])
].copy()
# 剔除 2017 年 9 月 1 日之前的订单（特征工程预热期），与 lgb_model.py 保持同一口径
data = data[data["order_time"] >= FULL_START_TIME].copy()
data = data.sort_values("order_time").reset_index(drop=True)

cutoff_time = data["order_time"].quantile(1 - CONFIG["test_size"])
TARGET_CENTER = float(np.median(data.loc[data["order_time"] < cutoff_time, "log_delivery_time"]))
print(f"PseudoHuber 中心化常数（与 lgb_model.py 一致）：{TARGET_CENTER:.6f}")

#%% 6. 加载模型并预测全量订单的运送周期
# 与 lgb_model.py 保持完全相同的特征集合与类别转换；estimated_delivery_time（原算法）仅作基准对比，不作为输入
drop_cols = ["order_time", "seller_id", "seller_city", "customer_city",
             "total_delivery_time", "log_delivery_time", "estimated_delivery_time"]
features = [c for c in data.columns if c not in drop_cols]

model_input = feature_data.copy()  # 保持特征表原始行序，即与 aligned_index 一一对应
for col in features:
    if pd.api.types.is_object_dtype(model_input[col]) or pd.api.types.is_string_dtype(model_input[col]):
        model_input[col] = model_input[col].astype("category")
model_input = model_input.replace([np.inf, -np.inf], np.nan)

X_all = model_input[features]
# 特征表中与订单数据逐行对应的下单时间，用于换算预计送达时间
order_time_aligned = df.loc[aligned_index, "下单时间"].reset_index(drop=True)

# L1 使用原始对数目标，PseudoHuber 系列使用中心化对数目标（与 lgb_model.py 的模型配置一致）
model_configs = {"L1": 0.0}
for weight in UNDER_WEIGHTS:
    model_configs[f"PseudoHuber_Under{weight:g}x"] = TARGET_CENTER

# 新变量先整体留空，再按对齐后的行位置回填
used_models = []
for model_name in model_configs:
    df[f"{model_name}预期总周期"] = np.nan
    df[f"{model_name}预计送达时间"] = pd.Series(pd.NA, index=df.index, dtype="object")

for model_name, center in model_configs.items():
    model_path = MODEL_DIR / f"{model_name}.txt"
    if not model_path.exists():
        print(f"未找到模型文件，已跳过：{model_path}")
        continue

    temp_path = TEMP_MODEL_DIR / f"{model_name}.txt"
    shutil.copy2(model_path, temp_path)  # 中文路径 -> 英文临时目录
    booster = lgb.Booster(model_file=str(temp_path))
    print(f"已加载模型：{model_path}")

    pred_log = booster.predict(X_all) + center
    pred_days = np.exp(np.clip(pred_log, -20, 20))

    # 换算预计送达时间，并反向修正预期总周期
    # 预测值是带小数的天数差（口径同 SQL 的 julianday 差值），但预计送达时间必须是某个日期的 00:00:00，
    # 因此先加到下单时间上，再向后取到下一个 00:00（如 2018-09-01 13:45:32 -> 2018-09-02 00:00:00）；
    # 取整后反过来用该整日期与下单时间求差值，得到与原算法「预期总周期」同口径的修正值。
    raw_eta = order_time_aligned + pd.to_timedelta(pred_days, unit="D")
    eta_date = raw_eta.dt.normalize() + pd.Timedelta(days=1)
    corrected_days = (eta_date - order_time_aligned).dt.total_seconds() / 86400

    df.loc[aligned_index, f"{model_name}预期总周期"] = corrected_days.round(2).to_numpy()
    df.loc[aligned_index, f"{model_name}预计送达时间"] = (
        eta_date.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
    )
    used_models.append(model_name)

    print(f"{model_name}：平均预期总周期 {df[f'{model_name}预期总周期'].mean():.3f} 天")

if not used_models:
    raise ValueError("没有可用的模型文件，请先在 model 目录下训练并保存模型。")

# 前 30 天预热期订单在 delay_distribution.R 中已被剔除，没有特征也算不出预测值；
# 8 月订单则属于特征工程预热期，也不纳入模型口径。
# 导出和后续统计都统一只用 2017-09-01 起的可预测订单，保证原算法与各模型在同一批样本上对比
has_feature = df[f"{used_models[0]}预期总周期"].notna()
has_prediction = has_feature & (df["下单时间"] >= FULL_START_TIME)
print(f"\n可预测订单：{int(has_prediction.sum())} 条 | 无特征订单：{int((~has_feature).sum())} 条 | "
      f"8 月预热订单：{int((has_feature & ~has_prediction).sum())} 条")
print(f"原算法平均预期总周期：{df.loc[has_prediction, '预期总周期'].mean():.3f} 天 | "
      f"真实平均总运送周期：{df.loc[has_prediction, '总运送周期'].mean():.3f} 天")

#%% 7. 输出预测数据（只保留有预测值的订单，表格内不留空值）
output_path = OUTPUT_DIR / "LightGBM预测数据.csv"
df.loc[has_prediction].to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\n预测数据已保存：{output_path}（{int(has_prediction.sum())} 行，不含无特征订单与 8 月预热订单）")

#%% 8. 构造预测结果长表，并切出与 lgb_model.py 一致的验证集
# 只取有预测值的订单，这样原算法与各模型的样本量完全一致
predict_df = df.loc[has_prediction]

# 误差 = 预期总周期 - 真实总运送周期；正数=按预期能提前送达，负数=实际会延迟
prediction_frames = [pd.DataFrame({
    "model": "Original_Algorithm",
    "actual_days": predict_df["总运送周期"].to_numpy(dtype=float),
    "pred_days": predict_df["预期总周期"].to_numpy(dtype=float),
})]
for model_name in used_models:
    prediction_frames.append(pd.DataFrame({
        "model": model_name,
        "actual_days": predict_df["总运送周期"].to_numpy(dtype=float),
        "pred_days": predict_df[f"{model_name}预期总周期"].to_numpy(dtype=float),
    }))

all_predictions = pd.concat(prediction_frames, ignore_index=True)

# lgb_model.py 只输出测试集（order_time 在后 test_size 比例）的预测值，
# 这里按同样的时间切分点把验证集挑出来，方便和 lgb_model.py 的结果逐一对照
is_validation = pd.Series(False, index=df.index)
is_validation.loc[aligned_index] = (order_time_aligned >= cutoff_time).to_numpy()
validation_mask = np.tile(is_validation.loc[has_prediction].to_numpy(), 1 + len(used_models))
validation_predictions = all_predictions.loc[validation_mask].copy()

print(f"\n验证集订单：{int(is_validation.sum())} 条 | 切分时间点：{cutoff_time}（与 lgb_model.py 的测试集一致）")

#%% 9. 评估与绘图函数：验证集与全量订单共用，指标口径和图形结构均与 lgb_model.py 一致
PCT_COLS = ["early_delivery_rate", "delay_rate", "exact_rate", "severe_overtime_rate"]


def display_name(model_name):
    """图例与表头里使用的显示名。"""
    return "原算法" if model_name == "Original_Algorithm" else model_name


def evaluate_predictions(predictions, scope_name):
    """计算预测精度、提前送达率与延迟风险（比例保存为 0~1）。"""
    metric_rows = []
    for model_name, group in predictions.groupby("model", sort=False):
        actual = group["actual_days"].to_numpy(dtype=float)
        pred = group["pred_days"].to_numpy(dtype=float)

        # 所有指标使用有效订单
        valid = np.isfinite(actual) & np.isfinite(pred) & (actual > 0) & (pred > 0)
        actual, pred = actual[valid], pred[valid]
        if len(actual) == 0:
            continue

        error = pred - actual
        metric_rows.append({
            "model": model_name, "sample_count": len(actual),
            "MAE_days": mean_absolute_error(actual, pred),
            "RMSE_days": np.sqrt(mean_squared_error(actual, pred)),
            "pred_mean_days": np.mean(pred), "actual_mean_days": np.mean(actual),
            "mean_error_days": np.mean(error),
            "early_delivery_rate": np.mean(error > 0), "delay_rate": np.mean(error < 0),
            "exact_rate": np.mean(np.isclose(pred, actual, rtol=1e-9, atol=1e-9)),
            "severe_overtime_rate": np.mean(error < -SEVERE_UNDER_DAYS),
        })

    if not metric_rows:
        raise ValueError(f"{scope_name}没有有效预测结果，请检查预测表。")
    metrics_df = pd.DataFrame(metric_rows).sort_values("MAE_days").reset_index(drop=True)

    # 仅在打印结果时转换成百分比
    result = metrics_df.copy()
    result[PCT_COLS] *= 100
    print(f"\n========== {scope_name}：预期总周期评估 ==========")
    print("提前送达率：预期总周期 > 真实总运送周期的订单占比。")
    print("延迟率：预期总周期 < 真实总运送周期的订单占比。")
    print(f"严重超时率：真实比预期多超过 {SEVERE_UNDER_DAYS:.1f} 天的订单占比。")
    print("比例单位为 %；时长和误差单位为天。")
    print(result.round(3).to_string(index=False))
    return metrics_df


def save_distribution_table(predictions, scope_name, file_name):
    """预期周期分布统计，表结构同 lgb_model.py 的 delivery_time_distribution.csv。"""
    distribution_rows = []

    def add_distribution_stats(name, values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        if len(values) == 0:
            return
        distribution_rows.append({
            "数据来源": name, "样本数": len(values),
            "均值": np.mean(values), "标准差": np.std(values), "最小值": np.min(values),
            "P25": np.quantile(values, 0.25), "中位数": np.median(values),
            "P75": np.quantile(values, 0.75), "P90": np.quantile(values, 0.90),
            "P95": np.quantile(values, 0.95), "最大值": np.max(values),
        })

    # 各分组的真实值列完全相同，取第一个分组即可代表真实周期
    first_group = next(iter(predictions.groupby("model", sort=False)))[1]
    add_distribution_stats(f"{scope_name}真实总运送周期", first_group["actual_days"].to_numpy(dtype=float))
    for model_name, group in predictions.groupby("model", sort=False):
        add_distribution_stats(f"{display_name(model_name)}预期总周期", group["pred_days"].to_numpy(dtype=float))

    distribution_df = pd.DataFrame(distribution_rows)
    print(f"\n========== {scope_name}：预期周期分布统计（单位：天） ==========")
    print(distribution_df.round(3).to_string(index=False))
    distribution_df.to_csv(OUTPUT_DIR / file_name, index=False, encoding="utf-8-sig")


def plot_distribution(predictions, scope_name, figure_name):
    """预期周期分布对比图：真实值 vs 原算法 vs 各模型。"""
    first_group = next(iter(predictions.groupby("model", sort=False)))[1]
    actual_days = first_group["actual_days"].to_numpy(dtype=float)
    actual_days = actual_days[np.isfinite(actual_days) & (actual_days > 0)]

    plt.figure(figsize=CONFIG["figure_size"])
    sns.kdeplot(actual_days, label="真实总运送周期", linewidth=2.8)
    for model_name, group in predictions.groupby("model", sort=False):
        pred = group["pred_days"].to_numpy(dtype=float)
        pred = pred[np.isfinite(pred) & (pred > 0)]
        if len(pred) > 1:
            sns.kdeplot(pred, label=display_name(model_name), linewidth=2)

    plt.title(f"{scope_name}：原算法、各模型预期总周期与真实周期分布对比")
    plt.xlabel("配送周期（天）")
    plt.ylabel("概率密度")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(ASSET_DIR / figure_name, dpi=CONFIG["figure_dpi"], bbox_inches="tight")
    plt.show()
    plt.close()


def plot_error_distribution(predictions, scope_name, figure_name):
    """预期周期误差分布图：预期总周期 - 真实总运送周期。"""
    plt.figure(figsize=CONFIG["figure_size"])
    for model_name, group in predictions.groupby("model", sort=False):
        actual = group["actual_days"].to_numpy(dtype=float)
        pred = group["pred_days"].to_numpy(dtype=float)
        valid = np.isfinite(actual) & np.isfinite(pred) & (actual > 0) & (pred > 0)
        errors = pred[valid] - actual[valid]
        if len(errors) > 1:
            sns.kdeplot(errors, label=display_name(model_name), linewidth=2)

    plt.axvline(0, color="red", linestyle="--", linewidth=1.8, label="零误差线")
    plt.title(f"{scope_name}：原算法与各模型的预期周期误差分布")
    plt.xlabel("预期总周期 - 真实总运送周期（天）")
    plt.ylabel("概率密度")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(ASSET_DIR / figure_name, dpi=CONFIG["figure_dpi"], bbox_inches="tight")
    plt.show()
    plt.close()


def plot_delay_rate(metrics_df, scope_name, figure_name):
    """不同算法的配送延迟率对比柱状图。"""
    model_names = [display_name(m) for m in metrics_df["model"].tolist()]
    delay_rates = metrics_df["delay_rate"].to_numpy() * 100

    fig, ax = plt.subplots(figsize=(13, 6), dpi=CONFIG["figure_dpi"])
    bars = ax.bar(model_names, delay_rates, width=0.65)

    ax.set_title(f"{scope_name}：不同算法的配送延迟率对比")
    ax.set_xlabel("模型名称")
    ax.set_ylabel("延迟率（%）")
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9)
    ax.set_ylim(0, max(delay_rates) * 1.15 if len(delay_rates) else 1)

    fig.tight_layout()
    delay_rate_fig_path = ASSET_DIR / figure_name
    fig.savefig(delay_rate_fig_path, dpi=CONFIG["figure_dpi"], bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"延迟率对比图已保存至：{delay_rate_fig_path.resolve()}")


def save_summary(metrics_df, scope_name, file_name):
    """汇总预测精度、提前送达率与延迟风险（比例转成 % 后保存）。"""
    mean_comparison = metrics_df[[
        "model", "sample_count", "MAE_days", "RMSE_days",
        "pred_mean_days", "actual_mean_days", "mean_error_days",
        "early_delivery_rate", "delay_rate", "exact_rate", "severe_overtime_rate",
    ]].copy()

    mean_comparison[PCT_COLS] *= 100

    print(f"\n========== {scope_name}：预测精度、提前送达率与延迟风险对比 ==========")
    print(mean_comparison.round(3).to_string(index=False))

    mean_comparison.to_csv(OUTPUT_DIR / file_name, index=False, encoding="utf-8-sig")


#%% 10. 验证集复核：与 lgb_model.py 相同时间切分的测试集
validation_metrics = evaluate_predictions(validation_predictions, "验证集")
validation_metrics.to_csv(OUTPUT_DIR / "val_test_timeout_model_comparison.csv", index=False, encoding="utf-8-sig")
save_distribution_table(validation_predictions, "验证集", "val_test_delivery_time_distribution.csv")
plot_distribution(validation_predictions, "验证集", "val_test_delivery_time_distribution.svg")
plot_error_distribution(validation_predictions, "验证集", "val_test_delivery_time_error_distribution.svg")
plot_delay_rate(validation_metrics, "验证集", "val_test_delivery_delay_rate_comparison.svg")
save_summary(validation_metrics, "验证集", "val_test_mean_duration_and_risk_comparison.csv")

#%% 11. 全量订单：预期总周期评估
metrics_df = evaluate_predictions(all_predictions, "全量订单")
metrics_df.to_csv(OUTPUT_DIR / "val_timeout_model_comparison.csv", index=False, encoding="utf-8-sig")

#%% 12. 全量订单的预期周期分布：各模型 vs 原算法 vs 真实值
save_distribution_table(all_predictions, "全量订单", "val_delivery_time_distribution.csv")
plot_distribution(all_predictions, "全量订单", "val_delivery_time_distribution.svg")

#%% 13. 全量订单预期误差分布图
plot_error_distribution(all_predictions, "全量订单", "val_delivery_time_error_distribution.svg")

#%% 14. 全量订单不同算法的延迟率对比
plot_delay_rate(metrics_df, "全量订单", "val_delivery_delay_rate_comparison.svg")

#%% 15. 全量订单汇总：预测精度、提前送达率与延迟风险
save_summary(metrics_df, "全量订单", "val_mean_duration_and_risk_comparison.csv")

print(f"\n预测数据保存目录：{OUTPUT_DIR.resolve()}")
print(f"图表保存目录：{ASSET_DIR.resolve()}")
