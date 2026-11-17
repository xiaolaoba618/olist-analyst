# %% 1. 导入库
"""
Transformer + VCN 验证脚本（对标 lgb_val.py）

职责：
  1) 复用 lgb_val.py 的行对齐规则，把特征表每一行还原到订单数据的原始行位置；
  2) 用训练脚本保存的 meta（时间桶原点、类别映射、标准化统计量）重建序列与静态特征；
  3) 加载 model/ 下各 TransCNN 权重，对全量订单预测对数配送周期；
  4) 按部署口径把预测天数换算成 00:00:00 的预计送达时间，再反算「预期总周期」；
  5) 输出 TransformerCNN预测数据.csv，并做验证集 / 全量两套评估表与图。

依赖顺序：Transformer_CNN_model.py（先训练并保存模型与 meta）-> 本脚本。
import 训练脚本只取公共函数，不会触发训练（训练主流程有 __main__ 保护）。
"""
import pandas as pd
import numpy as np
import torch
import joblib
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from data_prepare import df as feature_data
from Transformer_CNN_model import (
    CONFIG, BASE_DIR, OUTPUT_DIR, ASSET_DIR, MODEL_DIR, DEVICE, PREFIX, SNAP_DIM,
    build_sequence_features, build_static_features, normalize_apply, days_from_log,
    evaluate_predictions, save_distribution_table, plot_distribution,
    plot_error_distribution, plot_delay_rate, save_summary,
)

# %% 2. 全局配置
FULL_START_TIME = pd.Timestamp(CONFIG["full_start_date"])
META_PATH = OUTPUT_DIR / f"{PREFIX}meta.joblib"
if not META_PATH.exists():
    raise FileNotFoundError(f"缺少训练脚本产出的元信息，请先运行 Transformer_CNN_model.py：{META_PATH}")
meta = joblib.load(META_PATH)

print(f"计算设备：{DEVICE}")
print(f"序列参数：{meta['seq_steps']} 步 × {meta['bucket_days']} 天 | 原点 {meta['seq_t0']} | "
      f"模式 {meta['seq_mode']}")

# %% 3. 读取订单数据 + 行对齐（规则与 lgb_val.py 完全相同）
df = pd.read_csv(BASE_DIR / "data" / "订单数据.csv", encoding="utf-8-sig")
df["下单时间"] = pd.to_datetime(df["下单时间"], format="mixed")

warm_mask = df["下单时间"] >= df["下单时间"].min() + pd.Timedelta(days=30)
aligned_index = (
    df.loc[warm_mask]
    .sort_values(["发货州", "收货州", "下单时间", "seller_id"], kind="stable")
    .index
)
if len(aligned_index) != len(feature_data):
    raise ValueError(
        f"行数对不上：订单数据筛出 {len(aligned_index)} 行，特征表有 {len(feature_data)} 行，"
        "请检查 delay_distribution.R 是否重新运行过。")

# 关键列逐行校验，确保行序完全一致
check_pairs = [
    ("seller_id", "seller_id"), ("商品类型", "product_type"), ("付款方式", "payment_method"),
    ("发货州", "seller_state"), ("收货州", "customer_state"),
    ("总运送周期", "total_delivery_time"), ("预期总周期", "estimated_delivery_time"),
]
mismatch = 0
for raw_col, feat_col in check_pairs:
    left = pd.Series(df.loc[aligned_index, raw_col].to_numpy())
    right = pd.Series(feature_data[feat_col].to_numpy())
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        bad = ~np.isclose(left.astype(float), right.astype(float), atol=1e-6, equal_nan=True)
    else:
        bad = (left != right) & ~(left.isna() & right.isna())
    mismatch += int(np.asarray(bad).sum())
if mismatch > 0:
    raise ValueError(f"行对齐校验失败：{mismatch} 处关键列不一致。")

print(f"行对齐校验通过：{len(aligned_index)} 行逐行一致")

# %% 4. 用训练的 meta 重建全量序列与静态特征
feat = feature_data.copy()
feat["order_time"] = pd.to_datetime(feat["order_time"], errors="coerce")

# 注意：t0 必须沿用训练时的原点，否则时间桶编号整体偏移
seq_all, _, _, _ = build_sequence_features(
    feat, seq_steps=meta["seq_steps"], bucket_days=meta["bucket_days"],
    seq_mode=meta["seq_mode"], t0_override=meta["seq_t0"])
static_raw, _ = build_static_features(feat, cat_categories=meta["cat_categories"])

seq_all = normalize_apply(seq_all.reshape(-1, SNAP_DIM), meta["seq_stats"]).reshape(
    len(feat), meta["seq_steps"], SNAP_DIM)
static_all = normalize_apply(static_raw, meta["static_stats"])

order_time_aligned = df.loc[aligned_index, "下单时间"].reset_index(drop=True)
print(f"特征重建完成：序列 {seq_all.shape} | 静态 {static_all.shape}")


# %% 5. 批量预测函数
def predict_days(model, seq, static, batch_size=2048):
    model.eval()
    ds = TensorDataset(torch.tensor(seq, dtype=torch.float32),
                       torch.tensor(static, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False,
                        pin_memory=(DEVICE.type == "cuda"))
    outs = []
    with torch.no_grad():
        for sb, stb in loader:
            pred_log = model(sb.to(DEVICE), stb.to(DEVICE)).cpu().numpy()
            outs.append(days_from_log(pred_log))
    return np.concatenate(outs)


# %% 6. 加载各模型，预测并回填（换算 00:00 预计送达时间）
model_records = meta["model_records"]
used_models = []

for rec in model_records:
    name = rec["model"]
    weight_path = MODEL_DIR / rec["state_dict"]
    if not weight_path.exists():
        print(f"未找到权重文件，已跳过：{weight_path}")
        continue

    model = TransformerVCN(static_dim=static_all.shape[1]).to(DEVICE)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    pred_days = predict_days(model, seq_all, static_all)
    print(f"已加载 {name} | 原始预测均值 {np.mean(pred_days):.3f} 天")

    # 部署口径换算：预测天数加到下单时间 -> 向后取下一个 00:00 -> 反算天数差
    raw_eta = order_time_aligned + pd.to_timedelta(pd.Series(pred_days), unit="D")
    eta_date = raw_eta.dt.normalize() + pd.Timedelta(days=1)
    eta_date = eta_date.where(raw_eta.dt.normalize() < raw_eta, raw_eta.dt.normalize())
    corrected_days = ((eta_date - order_time_aligned).dt.total_seconds() / 86400).round(2)

    df.loc[aligned_index, f"{name}预期总周期"] = corrected_days.to_numpy()
    df.loc[aligned_index, f"{name}预计送达时间"] = (
        eta_date.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy())
    used_models.append(name)

if not used_models:
    raise ValueError("没有可用的模型权重，请先运行 Transformer_CNN_model.py。")

# 只保留 2017-09-01 起、有预测值的订单（与 lgb_val.py 口径一致）
has_prediction = df[f"{used_models[0]}预期总周期"].notna() & (df["下单时间"] >= FULL_START_TIME)
print(f"\n可预测订单：{int(has_prediction.sum())} 条 | "
      f"无特征订单：{int((~has_prediction & df[f'{used_models[0]}预期总周期'].isna()).sum())} 条 | "
      f"8 月预热订单：{int((df[f'{used_models[0]}预期总周期'].notna() & ~has_prediction).sum())} 条")

# %% 7. 输出全量预测数据表（对标 LightGBM预测数据.csv）
output_path = OUTPUT_DIR / CONFIG["tc_pred_filename"]
df.loc[has_prediction].to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"预测数据已保存：{output_path}（{int(has_prediction.sum())} 行）")

# %% 8. 构造长表 + 切出与训练脚本一致的验证集
predict_df = df.loc[has_prediction]
frames = [pd.DataFrame({
    "model": "Original_Algorithm",
    "actual_days": predict_df["总运送周期"].to_numpy(dtype=float),
    "pred_days": predict_df["预期总周期"].to_numpy(dtype=float),
})]
for name in used_models:
    frames.append(pd.DataFrame({
        "model": name,
        "actual_days": predict_df["总运送周期"].to_numpy(dtype=float),
        "pred_days": predict_df[f"{name}预期总周期"].to_numpy(dtype=float),
    }))
all_predictions = pd.concat(frames, ignore_index=True)

# 用 meta 中的切分点，保证与 Transformer_CNN_model.py 的测试集一致
cutoff_time = pd.Timestamp(meta["cutoff_time"])
is_val = df.loc[has_prediction, "下单时间"] >= cutoff_time
val_mask = np.tile(is_val.to_numpy(), 1 + len(used_models))
val_predictions = all_predictions.loc[val_mask].copy()
print(f"验证集订单：{int(is_val.sum())} 条 | 切分点 {cutoff_time}")

# %% 9. 验证集评估与图
VAL_TAG = "验证集"
val_metrics = evaluate_predictions(val_predictions, VAL_TAG)
val_metrics.to_csv(OUTPUT_DIR / f"{PREFIX}val_test_timeout_model_comparison.csv",
                   index=False, encoding="utf-8-sig")
save_distribution_table(val_predictions, VAL_TAG, f"{PREFIX}val_test_delivery_time_distribution.csv")
plot_distribution(val_predictions, VAL_TAG, f"{PREFIX}val_test_delivery_time_distribution.svg")
plot_error_distribution(val_predictions, VAL_TAG, f"{PREFIX}val_test_delivery_time_error_distribution.svg")
plot_delay_rate(val_metrics, VAL_TAG, f"{PREFIX}val_test_delivery_delay_rate_comparison.svg")
save_summary(val_metrics, VAL_TAG, f"{PREFIX}val_test_mean_duration_and_risk_comparison.csv")

# %% 10. 全量订单评估与图
FULL_TAG = "全量订单"
full_metrics = evaluate_predictions(all_predictions, FULL_TAG)
full_metrics.to_csv(OUTPUT_DIR / f"{PREFIX}val_timeout_model_comparison.csv",
                    index=False, encoding="utf-8-sig")
save_distribution_table(all_predictions, FULL_TAG, f"{PREFIX}val_delivery_time_distribution.csv")
plot_distribution(all_predictions, FULL_TAG, f"{PREFIX}val_delivery_time_distribution.svg")
plot_error_distribution(all_predictions, FULL_TAG, f"{PREFIX}val_delivery_time_error_distribution.svg")
plot_delay_rate(full_metrics, FULL_TAG, f"{PREFIX}val_delivery_delay_rate_comparison.svg")
save_summary(full_metrics, FULL_TAG, f"{PREFIX}val_mean_duration_and_risk_comparison.csv")

print(f"\n表格保存目录：{OUTPUT_DIR.resolve()}\n图表保存目录：{ASSET_DIR.resolve()}")
