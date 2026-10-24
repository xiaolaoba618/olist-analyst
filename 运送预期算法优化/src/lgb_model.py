#%% 1. 导入库
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_prepare import df, BASE_DIR

#%% 2. 全局配置：实验时主要修改这里
CONFIG = {
    # 随机性与数据切分
    "random_state": 42, "num_boost_round": 1000, "test_size": 0.20,
    # 非对称 Pseudo-Huber 损失
    "under_weights": [3.0, 5.0, 10.0, 20.0], "over_weight": 1.0,
    "delta": 0.2,  # delta 作用于对数配送时长空间
    # 超时风险评估
    "deadline_days": 10.0, "severe_under_days": 3.0,
    # LightGBM 参数
    "learning_rate": 0.03, "num_leaves": 31, "max_depth": -1, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l1": 0.0, "lambda_l2": 2.0, "min_sum_hessian_in_leaf": 1e-3, "num_threads": -1,
    # 图表设置
    "figure_size": (12, 6), "figure_dpi": 200,
    
    # 输出路径
    "model_dir": Path(__file__).parent / "model",
    "asset_dir": Path(BASE_DIR) / "assets",
    'report_dir': Path(BASE_DIR) / "report",
    'output_dir': Path(BASE_DIR) / "output",
}

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

RANDOM_STATE, NUM_BOOST_ROUND = CONFIG["random_state"], CONFIG["num_boost_round"]
UNDER_WEIGHTS, OVER_WEIGHT = CONFIG["under_weights"], CONFIG["over_weight"]
DELTA, DEADLINE_DAYS = CONFIG["delta"], CONFIG["deadline_days"]
SEVERE_UNDER_DAYS = CONFIG["severe_under_days"]

OUTPUT_DIR, ASSET_DIR = Path(CONFIG["output_dir"]), Path(CONFIG["asset_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ASSET_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path(CONFIG["model_dir"])
MODEL_DIR.mkdir(parents=True, exist_ok=True)

#%% 3. 数据准备
data = df.copy()
data["order_time"] = pd.to_datetime(data["order_time"], errors="coerce")

# 真实配送时间、对数目标和原算法预计时间均需有效
required_cols = ["order_time", "total_delivery_time", "log_delivery_time", "estimated_delivery_time"]
missing_cols = [col for col in required_cols if col not in data.columns]
if missing_cols:
    raise ValueError(f"原数据框缺少必要字段：{missing_cols}")

# 确保配送时长字段是数值类型
for col in ["total_delivery_time", "log_delivery_time", "estimated_delivery_time"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# 删除必要字段缺失的记录
data = data.dropna(subset=required_cols).copy()

# 真实配送时间和原算法预计时间必须为正
data = data[
    (data["total_delivery_time"] > 0)
    & (data["estimated_delivery_time"] > 0)
    & np.isfinite(data["log_delivery_time"])
    & np.isfinite(data["total_delivery_time"])
    & np.isfinite(data["estimated_delivery_time"])
].copy()
data = data.sort_values("order_time").reset_index(drop=True)

# 防止目标、高基数 ID 以及原算法预测值进入模型；estimated_delivery_time 仅用于基准对比，不作为训练特征
drop_cols = ["order_time", "seller_id", "seller_city", "customer_city",
             "total_delivery_time", "log_delivery_time", "estimated_delivery_time"]
features = [c for c in data.columns if c not in drop_cols]

# 字符串类别转为 category
for col in features:
    if pd.api.types.is_object_dtype(data[col]) or pd.api.types.is_string_dtype(data[col]):
        data[col] = data[col].astype("category")
data = data.replace([np.inf, -np.inf], np.nan)

print(f"数据总量：{len(data)} | 特征数量：{len(features)} | 目标变量：log_delivery_time")
print("原算法基准：estimated_delivery_time（天）")

#%% 4. 按时间切分训练集和测试集
cutoff_time = data["order_time"].quantile(1 - CONFIG["test_size"])
train = data[data["order_time"] < cutoff_time].copy()
test = data[data["order_time"] >= cutoff_time].copy()
if len(train) == 0 or len(test) == 0:
    raise ValueError("时间切分后训练集或测试集为空，请检查配置。")

X_train, y_train = train[features], train["log_delivery_time"].to_numpy(dtype=float)
X_test, y_test_days = test[features], test["total_delivery_time"].to_numpy(dtype=float)

# 原算法在相同测试集上的预计配送时间
original_estimated_days = test["estimated_delivery_time"].to_numpy(dtype=float)

# 真实超时：实际配送时间超过原算法预计时间
actual_late = y_test_days > original_estimated_days

print(f"测试集原算法实际超时率：{actual_late.mean():.2%}")

print(f"训练集：{len(train)} | 测试集：{len(test)} | 测试集占比：{len(test) / len(data):.1%}")

dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature="auto", free_raw_data=False)

#%% 5. 自定义非对称 Pseudo-Huber 损失
def make_pseudo_huber(under_weight=3.0, over_weight=1.0, delta=0.2):
    def objective(preds, train_data):
        error = preds - train_data.get_label()
        # 预测偏低（低估）时采用更大的惩罚权重
        weights = np.where(error < 0, under_weight, over_weight)
        denom = np.sqrt(1.0 + (error / delta) ** 2)
        grad = weights * error / denom
        hess = np.maximum(weights / denom ** 3, 1e-6)
        return grad, hess
    return objective

# 中心化标签，提高自定义目标的训练稳定性
TARGET_CENTER = float(np.median(y_train))
y_train_centered = y_train - TARGET_CENTER
dtrain_custom = lgb.Dataset(X_train, label=y_train_centered, categorical_feature="auto", free_raw_data=False)

# %% 6. 训练模型：原算法基准 + L1 + 非对称模型
from pathlib import Path

# 创建模型保存目录
MODEL_DIR = OUTPUT_DIR / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

base_params = {k: CONFIG[k] for k in [
    "learning_rate", "num_leaves", "max_depth", "min_data_in_leaf",
    "feature_fraction", "bagging_fraction", "bagging_freq",
    "lambda_l1", "lambda_l2", "min_sum_hessian_in_leaf", "num_threads",
]}
base_params.update({"verbosity": -1, "seed": RANDOM_STATE})

model_configs = {"L1": {"train_set": dtrain, "params": {"objective": "regression_l1"}, "center": 0.0}}

# 按配置自动生成非对称模型
for weight in UNDER_WEIGHTS:
    model_configs[f"PseudoHuber_Under{weight:g}x"] = {
        "train_set": dtrain_custom,
        "params": {
            "objective": make_pseudo_huber(under_weight=weight, over_weight=OVER_WEIGHT, delta=DELTA),
            "boost_from_average": False,
        },
        "center": TARGET_CENTER,
    }

prediction_frames = []
trained_models = {}  # 保存各个训练好的模型对象

# 原算法基准
prediction_frames.append(pd.DataFrame({
    "model": "Original_Algorithm", "actual_days": y_test_days,
    "pred_days": original_estimated_days, "original_estimated_days": original_estimated_days,
}))

# 训练、保存并预测各个模型
for model_name, model_config in model_configs.items():
    print(f"正在训练模型：{model_name}")
    model = lgb.train(params={**base_params, **model_config["params"]},
                      train_set=model_config["train_set"], num_boost_round=NUM_BOOST_ROUND)
    trained_models[model_name] = model  # 保存训练好的 Booster 对象

    # 保存模型文件
    model_path = MODEL_DIR / f"{model_name}.txt"


    TEMP_MODEL_DIR = Path(r"G:\lgb_models_temp")
    TEMP_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    MODEL_DIR = Path(OUTPUT_DIR) / "saved_models"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    temp_path = TEMP_MODEL_DIR / f"{model_name}.txt"
    final_path = MODEL_DIR / f"{model_name}.txt"

    # LightGBM 先保存到英文路径
    model.save_model(str(temp_path))

    # Python 再复制到中文目录
    shutil.copy2(temp_path, final_path)

    print(f"模型已保存：{final_path}")

    # 预测
    pred_log = model.predict(X_test) + model_config["center"]
    pred_days = np.exp(np.clip(pred_log, -20, 20))
    prediction_frames.append(pd.DataFrame({
        "model": model_name, "actual_days": y_test_days,
        "pred_days": pred_days, "original_estimated_days": original_estimated_days,
    }))

# 合并所有模型的预测结果
all_predictions = pd.concat(prediction_frames, ignore_index=True)
all_predictions.to_csv(OUTPUT_DIR / "timeout_model_predictions.csv", index=False, encoding="utf-8-sig")

print(f"\n所有模型已保存至：{MODEL_DIR}")

# %% 7. 评估配送时长预测：预测精度、提前送达率与延迟风险

# 严重超时阈值：预测时长比真实时长少超过 7 天
SEVERE_UNDER_DAYS = 7.0
metric_rows = []

for model_name, group in all_predictions.groupby("model", sort=False):
    actual = group["actual_days"].to_numpy(dtype=float)
    pred = group["pred_days"].to_numpy(dtype=float)

    # 所有指标使用有效订单
    valid = np.isfinite(actual) & np.isfinite(pred) & (actual > 0) & (pred > 0)
    actual, pred = actual[valid], pred[valid]
    if len(actual) == 0:
        continue

    # 误差 = 预测时长 - 真实时长；正数=实际提前送达，负数=实际配送延迟
    error = pred - actual

    # 提前送达率 / 延迟率 / 相等率 / 严重超时率
    early_delivery_rate = np.mean(error > 0)
    delay_rate = np.mean(error < 0)
    exact_rate = np.mean(np.isclose(pred, actual, rtol=1e-9, atol=1e-9))
    severe_overtime_rate = np.mean(error < -SEVERE_UNDER_DAYS)

    metric_rows.append({
        "model": model_name, "sample_count": len(actual),
        "MAE_days": mean_absolute_error(actual, pred),
        "RMSE_days": np.sqrt(mean_squared_error(actual, pred)),
        "pred_mean_days": np.mean(pred), "actual_mean_days": np.mean(actual),
        "mean_error_days": np.mean(error),
        "early_delivery_rate": early_delivery_rate, "delay_rate": delay_rate,
        "exact_rate": exact_rate, "severe_overtime_rate": severe_overtime_rate,
    })

metrics_df = pd.DataFrame(metric_rows)
if metrics_df.empty:
    raise ValueError("没有有效预测结果，请检查 all_predictions。")
metrics_df = metrics_df.sort_values("MAE_days").reset_index(drop=True)

# 保存原始比例（0~1）
metrics_df.to_csv(OUTPUT_DIR / "timeout_model_comparison.csv", index=False, encoding="utf-8-sig")

# 仅在打印结果时转换成百分比
pct_cols = ["early_delivery_rate", "delay_rate", "exact_rate", "severe_overtime_rate"]
result = metrics_df.copy()
result[pct_cols] *= 100

print("\n========== 配送时长预测评估 ==========")
print("提前送达率：预测时长 > 真实时长的订单占有效订单比例。")
print("延迟率：预测时长 < 真实时长的订单占有效订单比例。")
print("相等率：预测时长与真实时长近似相等的订单占比。")
print(f"严重超时率：实际配送时长比预测时长多超过 {SEVERE_UNDER_DAYS:.1f} 天的订单占比。")
print("提前送达率 + 延迟率 + 相等率应约等于 100%。")
print("比例单位为 %；时长和误差单位为天。")
print(result.round(3).to_string(index=False))

# 单独核对原算法
original_rows = metrics_df[metrics_df["model"] == "Original_Algorithm"]
if not original_rows.empty:
    original = original_rows.iloc[0]
    print("\n========== 原算法逐单核对 ==========")
    print(f"有效订单数：{int(original['sample_count'])}")
    print(f"提前送达率：{original['early_delivery_rate'] * 100:.3f}%")
    print(f"延迟率：{original['delay_rate'] * 100:.3f}%")
    print(f"相等率：{original['exact_rate'] * 100:.3f}%")
    print("三项合计：" f"{(original['early_delivery_rate'] + original['delay_rate'] + original['exact_rate']) * 100:.3f}%")
    print(f"严重超时率：{original['severe_overtime_rate'] * 100:.3f}%")

# %% 8. 配送周期分布：各算法预测值 vs 真实值
actual_days = np.asarray(y_test_days, dtype=float)
actual_days = actual_days[np.isfinite(actual_days) & (actual_days > 0)]
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

add_distribution_stats("测试集真实配送周期", actual_days)
for model_name, group in all_predictions.groupby("model", sort=False):
    add_distribution_stats(f"{model_name}_预测值", group["pred_days"].to_numpy(dtype=float))

distribution_df = pd.DataFrame(distribution_rows)
print("\n========== 配送周期分布统计（单位：天） ==========")
print(distribution_df.round(3).to_string(index=False))
distribution_df.to_csv(OUTPUT_DIR / "delivery_time_distribution.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=CONFIG["figure_size"])
sns.kdeplot(actual_days, label="真实配送周期", linewidth=2.8)
for model_name, group in all_predictions.groupby("model", sort=False):
    pred = group["pred_days"].to_numpy(dtype=float)
    pred = pred[np.isfinite(pred) & (pred > 0)]
    if len(pred) > 1:
        label = "原算法" if model_name == "Original_Algorithm" else model_name
        sns.kdeplot(pred, label=label, linewidth=2)

plt.title("原算法、各模型与真实配送周期分布对比")
plt.xlabel("配送周期（天）")
plt.ylabel("概率密度")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(ASSET_DIR / "delivery_time_distribution.svg", dpi=CONFIG["figure_dpi"], bbox_inches="tight")
plt.show()
plt.close()

# %% 9. 预测误差分布图
plt.figure(figsize=CONFIG["figure_size"])
for model_name, group in all_predictions.groupby("model", sort=False):
    actual = group["actual_days"].to_numpy(dtype=float)
    pred = group["pred_days"].to_numpy(dtype=float)
    valid = np.isfinite(actual) & np.isfinite(pred) & (actual > 0) & (pred > 0)
    errors = pred[valid] - actual[valid]
    if len(errors) > 1:
        label = "原算法" if model_name == "Original_Algorithm" else model_name
        sns.kdeplot(errors, label=label, linewidth=2)

plt.axvline(0, color="red", linestyle="--", linewidth=1.8, label="零误差线")
plt.title("原算法与各模型的配送时长预测误差分布")
plt.xlabel("预测值 - 真实值（天）")
plt.ylabel("概率密度")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(ASSET_DIR / "delivery_time_error_distribution.svg", dpi=CONFIG["figure_dpi"], bbox_inches="tight")
plt.show()
plt.close()

# %% 10. 不同算法的延迟率对比
plot_df = metrics_df.copy()
model_names = plot_df["model"].tolist()
delay_rates = plot_df["delay_rate"].to_numpy() * 100

fig, ax = plt.subplots(figsize=(13, 6), dpi=CONFIG["figure_dpi"])
bars = ax.bar(model_names, delay_rates, width=0.65)

ax.set_title("不同算法的配送延迟率对比")
ax.set_xlabel("模型名称")
ax.set_ylabel("延迟率（%）")
ax.set_xticks(np.arange(len(model_names)))
ax.set_xticklabels(model_names, rotation=25, ha="right")
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9)
ax.set_ylim(0, max(delay_rates) * 1.15 if len(delay_rates) else 1)

fig.tight_layout()
delay_rate_fig_path = ASSET_DIR / "delivery_delay_rate_comparison.svg"
fig.savefig(delay_rate_fig_path, dpi=CONFIG["figure_dpi"], bbox_inches="tight")
plt.show()
plt.close(fig)

print(f"延迟率对比图已保存至：{delay_rate_fig_path.resolve()}")

# %% 11. 汇总预测精度、提前送达率与延迟风险
mean_comparison = metrics_df[[
    "model", "sample_count", "MAE_days", "RMSE_days",
    "pred_mean_days", "actual_mean_days", "mean_error_days",
    "early_delivery_rate", "delay_rate", "exact_rate", "severe_overtime_rate",
]].copy()

pct_cols = ["early_delivery_rate", "delay_rate", "exact_rate", "severe_overtime_rate"]
mean_comparison[pct_cols] *= 100

print("\n========== 预测精度、提前送达率与延迟风险对比 ==========")
print(mean_comparison.round(3).to_string(index=False))

mean_comparison.to_csv(OUTPUT_DIR / "mean_duration_and_risk_comparison.csv", index=False, encoding="utf-8-sig")

print(f"\n表格与预测数据保存目录：{OUTPUT_DIR.resolve()}")
print(f"图表保存目录：{ASSET_DIR.resolve()}")




