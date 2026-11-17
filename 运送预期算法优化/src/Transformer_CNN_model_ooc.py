# %% 1. 导入库
"""
Transformer + VCN 折外验证脚本（对标 oof_val.py）

职责：
  1) 时间序列前向链 K 折：每折只用「该折之前」的订单训练，预测该折，得到样本外 OOF 预测；
  2) 与全量（含样本内乐观）、时间外测试期一起做三口径对照；
  3) 做最终模型 vs 折外模型、按月漂移、按实际时长分桶等诊断；
  4) 导出 tcn_ooc_predictions.csv（row_pos 直接对应 TransformerCNN预测数据.csv 的行号），
     供下游 GMV 分析零歧义复用。

依赖顺序：Transformer_CNN_model.py -> Transformer_CNN_model_val.py -> 本脚本。
"""
import pandas as pd
import numpy as np
import torch
import joblib
from pathlib import Path

from data_prepare import df as feature_data
from Transformer_CNN_model import (
    CONFIG, BASE_DIR, OUTPUT_DIR, ASSET_DIR, DEVICE, PREFIX, SNAP_DIM,
    build_sequence_features, build_static_features, normalize_fit_transform, normalize_apply,
    train_one_model, days_from_log, display_name, evaluate_predictions, plot_distribution,
    plot_error_distribution, plot_delay_rate, save_summary,
)

# %% 2. 全局配置
plt_needed = True  # 保留开关，便于无图形环境下跳过绘图
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FULL_START_TIME = pd.Timestamp(CONFIG["full_start_date"])
SEVERE_UNDER_DAYS = CONFIG["severe_under_days"]
META_PATH = OUTPUT_DIR / f"{PREFIX}meta.joblib"
FULL_PRED_PATH = OUTPUT_DIR / CONFIG["tc_pred_filename"]
if not META_PATH.exists() or not FULL_PRED_PATH.exists():
    raise FileNotFoundError(
        "请先依次运行 Transformer_CNN_model.py 与 Transformer_CNN_model_val.py："
        f"{META_PATH} / {FULL_PRED_PATH}")
meta = joblib.load(META_PATH)
full_pred = pd.read_csv(FULL_PRED_PATH, encoding="utf-8-sig")
full_pred["下单时间"] = pd.to_datetime(full_pred["下单时间"], format="mixed")

N_FOLDS = CONFIG["tc_oof_folds"]
OOF_EPOCHS = CONFIG["tc_oof_epochs"]
model_records = meta["model_records"]
MODEL_NAMES = [r["model"] for r in model_records]
print(f"计算设备：{DEVICE} | 折数 {N_FOLDS} | 每折轮数 {OOF_EPOCHS} | 模型 {MODEL_NAMES}")

# %% 3. 重建特征表（保持与 val 完全相同的行序：feature_data 原行序 = aligned_index 顺序）
feat = feature_data.copy()
feat["order_time"] = pd.to_datetime(feat["order_time"], errors="coerce")
feat = feat[feat["order_time"] >= FULL_START_TIME].reset_index(drop=True)
order_time = feat["order_time"].reset_index(drop=True)

seq_raw, _, _, _ = build_sequence_features(
    feat, seq_steps=meta["seq_steps"], bucket_days=meta["bucket_days"],
    seq_mode=meta["seq_mode"], t0_override=meta["seq_t0"])
static_raw, _ = build_static_features(feat, cat_categories=meta["cat_categories"])
y_log = feat["log_delivery_time"].to_numpy(dtype=float)
valid_target = np.isfinite(y_log) & (y_log != 0)
N = len(feat)
print(f"折外数据：{N} 行 | 时间范围 {order_time.min()} ~ {order_time.max()}")

# %% 4. 行号映射：特征表行 -> val 预测表行（供导出 row_pos）
# feature_data 第 i 行对应订单行 aligned_index[i]；val 预测表按订单原始行号升序输出。
orders = pd.read_csv(BASE_DIR / "data" / "订单数据.csv", encoding="utf-8-sig")
orders["下单时间"] = pd.to_datetime(orders["下单时间"], format="mixed")
warm_mask = orders["下单时间"] >= orders["下单时间"].min() + pd.Timedelta(days=30)
keep_index = orders.index[warm_mask & (orders["下单时间"] >= FULL_START_TIME)]
aligned_index = (orders.loc[warm_mask]
                 .sort_values(["发货州", "收货州", "下单时间", "seller_id"], kind="stable").index)
aligned_full = aligned_index[(orders.loc[aligned_index, "下单时间"] >= FULL_START_TIME).to_numpy()]
if len(aligned_full) != N:
    raise ValueError(f"行号映射长度不符：{len(aligned_full)} vs {N}")
row_pos = pd.Series(np.arange(len(keep_index)), index=keep_index).loc[aligned_full].to_numpy()

# %% 5. 时间序列前向链 K 折：只训练「该折之前」的数据
# 切折按 order_time 排序后的位置；但每折内部仍用特征表原始下标去索引矩阵。
order_sorted_idx = np.argsort(order_time.to_numpy(), kind="stable")
folds = np.array_split(order_sorted_idx, N_FOLDS)
oof_log = {name: np.full(N, np.nan) for name in MODEL_NAMES}

for k in range(1, N_FOLDS):
    train_idx = np.concatenate(folds[:k])
    train_idx = train_idx[valid_target[train_idx]]      # 目标缺失/异常的行不参与训练
    valid_idx = folds[k]
    valid_idx = valid_idx[valid_target[valid_idx]]
    print(f"\n===== 折 {k}/{N_FOLDS - 1} | 训练 {len(train_idx)} 条 | 验证 {len(valid_idx)} 条 =====")

    # 每折独立 fit 标准化，杜绝验证期信息泄漏
    (seq_tr_flat, seq_va_flat), _ = normalize_fit_transform(
        seq_raw[train_idx].reshape(-1, SNAP_DIM),
        seq_raw[valid_idx].reshape(-1, SNAP_DIM))
    seq_tr = seq_tr_flat.reshape(len(train_idx), meta["seq_steps"], SNAP_DIM)
    seq_va = seq_va_flat.reshape(len(valid_idx), meta["seq_steps"], SNAP_DIM)
    (static_tr, static_va), _ = normalize_fit_transform(
        static_raw[train_idx], static_raw[valid_idx])

    for rec in model_records:
        name = rec["model"]
        _, pred_log = train_one_model(
            name, rec["loss"], rec["under_weight"], rec["center"],
            seq_tr, static_tr, y_log[train_idx] - rec["center"], seq_va, static_va,
            static_dim=static_raw.shape[1], epochs=OOF_EPOCHS)
        oof_log[name][valid_idx] = pred_log

# OOF 天数 -> 00:00 取整 -> 反算预期总周期（部署口径，与 val 脚本一致）
OOF_DAYS = {}
for name in MODEL_NAMES:
    pred_days = days_from_log(oof_log[name])
    raw_eta = order_time + pd.to_timedelta(pd.Series(pred_days), unit="D")
    eta_date = raw_eta.dt.normalize() + pd.Timedelta(days=1)
    eta_date = eta_date.where(raw_eta.dt.normalize() < raw_eta, raw_eta.dt.normalize())
    OOF_DAYS[name] = ((eta_date - order_time).dt.total_seconds() / 86400).round(2).to_numpy()

oof_cols = [f"{m}_oof_预期总周期" for m in MODEL_NAMES]
oof_frame = pd.DataFrame({f"{m}_oof_预期总周期": OOF_DAYS[m] for m in MODEL_NAMES})
valid_mask = oof_frame.notna().all(axis=1).to_numpy()
print(f"\n折外有效订单：{int(valid_mask.sum())} 条（最早一段 {int((~valid_mask).sum())} 条无更早数据可训练）")

# %% 6. 三口径：全量 / 时间外测试期 / 折外
actual_all = full_pred["总运送周期"].to_numpy(dtype=float)
base_all = full_pred["预期总周期"].to_numpy(dtype=float)
full_models = {m: full_pred[f"{m}预期总周期"].to_numpy(dtype=float) for m in MODEL_NAMES}
cutoff_time = pd.Timestamp(meta["cutoff_time"])
test_mask = (full_pred["下单时间"] >= cutoff_time).to_numpy()

oof_models = {}
for m in MODEL_NAMES:
    arr = np.full(len(full_pred), np.nan)
    arr[row_pos[valid_mask]] = OOF_DAYS[m][valid_mask]
    oof_models[m] = arr
oof_scope = np.zeros(len(full_pred), dtype=bool)
oof_scope[row_pos[valid_mask]] = True

SCOPE_SRC = {"全量": full_models, "时间外测试期": full_models, "折外": oof_models}
SCOPE_MASK = {"全量": np.ones(len(full_pred), dtype=bool),
              "时间外测试期": test_mask, "折外": oof_scope}


def scope_frame(scope):
    mask = SCOPE_MASK[scope]
    mp = SCOPE_SRC[scope]
    return actual_all[mask], {m: mp[m][mask] for m in mp}, base_all[mask]


# 汇总表
summary_rows = []
for scope in ["全量", "时间外测试期", "折外"]:
    a, mp, b = scope_frame(scope)
    m = evaluate_predictions(pd.concat([
        pd.DataFrame({"model": "Original_Algorithm", "actual_days": a, "pred_days": b})] +
        [pd.DataFrame({"model": name, "actual_days": a, "pred_days": mp[name]}) for name in mp],
        ignore_index=True), scope).set_index("model")
    row = {"口径": scope, "样本数": int(m["sample_count"].iloc[0]),
           "原算法延迟率": m.loc["Original_Algorithm", "delay_rate"],
           "真实均值": m["actual_mean_days"].iloc[0]}
    for name in MODEL_NAMES:
        if name in m.index:
            row[f"{name}_延迟率"] = m.loc[name, "delay_rate"]
            row[f"{name}_MAE"] = m.loc[name, "MAE_days"]
            row[f"{name}_平均承诺"] = m.loc[name, "pred_mean_days"]
    summary_rows.append(row)

scope_summary = pd.DataFrame(summary_rows)
scope_summary.to_csv(OUTPUT_DIR / f"{PREFIX}ooc_scope_summary.csv", index=False, encoding="utf-8-sig")
print("\n========== 三口径延迟率对照（%）==========")
show = scope_summary.copy()
for c in [c for c in show.columns if "延迟率" in c]:
    show[c] = (show[c] * 100).round(2)
print(show.round(3).to_string(index=False))

# 折外口径完整评估
a_oof, mp_oof, b_oof = scope_frame("折外")
oof_pred_frame = pd.concat([
    pd.DataFrame({"model": "Original_Algorithm", "actual_days": a_oof, "pred_days": b_oof})] +
    [pd.DataFrame({"model": name, "actual_days": a_oof, "pred_days": mp_oof[name]})
     for name in MODEL_NAMES], ignore_index=True)
oof_metrics = evaluate_predictions(oof_pred_frame, "折外订单").sort_values("MAE_days").reset_index(drop=True)
oof_metrics.to_csv(OUTPUT_DIR / f"{PREFIX}ooc_timeout_model_comparison.csv",
                   index=False, encoding="utf-8-sig")
save_summary(oof_metrics, "折外订单", f"{PREFIX}ooc_mean_duration_and_risk_comparison.csv")

# %% 7. 折外三张图
plot_distribution(oof_pred_frame, "折外订单", f"{PREFIX}ooc_delivery_time_distribution.svg")
plot_error_distribution(oof_pred_frame, "折外订单", f"{PREFIX}ooc_delivery_time_error_distribution.svg")
plot_delay_rate(oof_metrics, "折外订单", f"{PREFIX}ooc_delivery_delay_rate_comparison.svg")

# %% 8. 诊断一：同批订单，最终模型 vs 折外模型
rows = []
for name in MODEL_NAMES:
    fin = full_models[name][oof_scope]
    out = oof_models[name][oof_scope]
    ok = np.isfinite(fin) & np.isfinite(out) & (a_oof > 0)
    f, o, aa = fin[ok], out[ok], a_oof[ok]
    rows.append({
        "模型": name,
        "最终_平均承诺": np.mean(f), "最终_延迟率": np.mean(aa > f), "最终_MAE": np.mean(np.abs(f - aa)),
        "折外_平均承诺": np.mean(o), "折外_延迟率": np.mean(aa > o), "折外_MAE": np.mean(np.abs(o - aa)),
        "承诺均值差": np.mean(o) - np.mean(f),
        "两模型预测相关系数": np.corrcoef(f, o)[0, 1],
    })
in_vs_out = pd.DataFrame(rows)
in_vs_out.to_csv(OUTPUT_DIR / f"{PREFIX}ooc_in_vs_out_comparison.csv", index=False, encoding="utf-8-sig")
print("\n========== 同批折外订单：最终模型 vs 折外模型 ==========")
print(in_vs_out.round(3).to_string(index=False))

# %% 9. 诊断二：按月漂移
month = full_pred["下单时间"].dt.to_period("M")
drift_rows = []
for m, idx in full_pred.groupby(month).groups.items():
    ix = np.asarray(idx)
    if not oof_scope[ix].any():
        continue
    row = {"月份": str(m), "订单数": len(ix), "实际均值": np.mean(actual_all[ix]),
           "原算法承诺": np.mean(base_all[ix])}
    for name in MODEL_NAMES:
        row[f"{name}_最终承诺"] = np.mean(full_models[name][ix])
        om = oof_models[name][ix][oof_scope[ix]]
        row[f"{name}_折外承诺"] = np.mean(om) if len(om) else np.nan
        row[f"{name}_折外延迟率"] = np.mean(actual_all[ix][oof_scope[ix]] > om) if len(om) else np.nan
    drift_rows.append(row)
monthly = pd.DataFrame(drift_rows)
monthly.to_csv(OUTPUT_DIR / f"{PREFIX}ooc_monthly_drift.csv", index=False, encoding="utf-8-sig")
print("\n========== 按月：真实配送时长漂移 ==========")
print(f"实际均值最高 {monthly.loc[monthly['实际均值'].idxmax(), '月份']} "
      f"{monthly['实际均值'].max():.2f} 天 | 最低 "
      f"{monthly.loc[monthly['实际均值'].idxmin(), '月份']} {monthly['实际均值'].min():.2f} 天")

# %% 10. 诊断三：按实际时长分桶（长尾识别能力）
bins = [0, 5, 10, 15, 20, 25, 30, 40, 60, np.inf]
labels = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-40", "40-60", "60+"]
sel_model = MODEL_NAMES[-2] if len(MODEL_NAMES) >= 2 else MODEL_NAMES[0]  # 默认看 20x
bucket_rows = []
for label, (lo, hi) in zip(labels, zip(bins[:-1], bins[1:])):
    sel = (a_oof >= lo) & (a_oof < hi)
    if sel.sum() < 30:
        continue
    row = {"实际时长区间": label, "订单数": int(sel.sum()), "实际均值": np.mean(a_oof[sel]),
           "原算法承诺": np.mean(b_oof[sel])}
    for name in MODEL_NAMES:
        row[f"{name}_折外承诺"] = np.mean(mp_oof[name][sel])
        row[f"{name}_折外延迟率"] = np.mean(mp_oof[name][sel] < a_oof[sel])
    bucket_rows.append(row)
buckets = pd.DataFrame(bucket_rows)
buckets.to_csv(OUTPUT_DIR / f"{PREFIX}ooc_extreme_diagnosis.csv", index=False, encoding="utf-8-sig")

if plt_needed and len(buckets):
    fig, ax = plt.subplots(figsize=CONFIG["figure_size"], dpi=CONFIG["figure_dpi"])
    x = np.arange(len(buckets))
    ax.plot(x, buckets["实际均值"], "k--", marker="o", linewidth=2, label="实际时长（理想参照）")
    ax.plot(x, buckets[f"{sel_model}_折外承诺"], marker="^", linewidth=2, label=f"{sel_model}·折外")
    ax.plot(x, buckets["原算法承诺"], ":", marker="v", linewidth=2, color="gray", label="原算法承诺")
    ax.set_xticks(x)
    ax.set_xticklabels(buckets["实际时长区间"])
    ax.set_xlabel("实际总运送周期区间（天）")
    ax.set_ylabel("平均承诺天数")
    ax.set_title("各实际时长区间下的平均承诺：折外模型对长尾订单的识别能力")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSET_DIR / f"{PREFIX}ooc_extreme_bucket.png", dpi=CONFIG["figure_dpi"],
                bbox_inches="tight")
    plt.show()
    plt.close(fig)

p99 = np.quantile(a_oof, 0.99)
extreme = a_oof > p99
print(f"\n实际时长 P99 = {p99:.2f} 天；超 P99 订单 {int(extreme.sum())} 单，"
      f"最长 {a_oof.max():.1f} 天")
print(f"这 {int(extreme.sum())} 单在 {sel_model} 折外口径下的延迟率："
      f"{np.mean(mp_oof[sel_model][extreme] < a_oof[extreme]) * 100:.2f}%")

# %% 11. 导出折外预测表（row_pos 直接对齐 TransformerCNN预测数据.csv）
export = pd.DataFrame({"row_pos": row_pos[valid_mask]})
for name in MODEL_NAMES:
    export[f"{name}_ooc_预期总周期"] = OOF_DAYS[name][valid_mask]
export = export.sort_values("row_pos").reset_index(drop=True)
export_path = OUTPUT_DIR / f"{PREFIX}ooc_predictions.csv"
export.to_csv(export_path, index=False, encoding="utf-8-sig")
print(f"\n折外预测值已导出：{export_path}（{len(export)} 行）")
print(f"表格保存目录：{OUTPUT_DIR.resolve()}\n图表保存目录：{ASSET_DIR.resolve()}")
