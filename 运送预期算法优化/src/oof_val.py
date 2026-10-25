#%% 1. 导入库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config import CONFIG

#%% 2. 全局配置：与 lgb_model.py / lgb_val.py 保持一致
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

UNDER_WEIGHTS = CONFIG["under_weights"]
# 严重超时阈值（天）
SEVERE_UNDER_DAYS = CONFIG["severe_under_days"]
# 全量口径起点：2017 年 8 月数据只用于初期构造滑动窗口等特征工程，不进入模型与统计
FULL_START_TIME = pd.Timestamp(CONFIG["full_start_date"])
# 非对称 Pseudo-Huber 的 delta 与低估惩罚倍率，用于分析极端离群值的惩罚行为
DELTA, OVER_WEIGHT = CONFIG["delta"], CONFIG["over_weight"]

BASE_DIR = CONFIG["base_dir"]
OUTPUT_DIR, ASSET_DIR = Path(CONFIG["output_dir"]), Path(CONFIG["asset_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ASSET_DIR.mkdir(parents=True, exist_ok=True)

# 四个模型的显示名（与 lgb_val.py 的列名一致）
MODEL_NAMES = ["L1"] + [f"PseudoHuber_Under{w:g}x" for w in UNDER_WEIGHTS]

#%% 3. 读取 lgb_val.py 的全量预测表与 lgb_model.py 的折外预测表
# 依赖顺序：lgb_model.py（训练 + 产出 OOF） -> lgb_val.py（回填全量） -> 本脚本
pred_path = OUTPUT_DIR / "LightGBM预测数据.csv"
oof_path = OUTPUT_DIR / "oof_predictions.csv"
if not pred_path.exists() or not oof_path.exists():
    raise FileNotFoundError(
        f"缺少输入文件，请先依次运行 lgb_model.py 与 lgb_val.py：{pred_path} / {oof_path}"
    )

pred = pd.read_csv(pred_path, encoding="utf-8-sig")
oof_raw = pd.read_csv(oof_path, encoding="utf-8-sig")

pred["下单时间"] = pd.to_datetime(pred["下单时间"], format="mixed")
print(f"全量预测表：{len(pred)} 行 | 折外预测表：{len(oof_raw)} 行")

#%% 4. 行对齐：折外预测表与全量预测表的行序不一致
# oof_predictions.csv 按 order_time 排序，但同一秒内下单的订单内部顺序与预测表不同（461 行互串），
# _row_id 记录的是预热过滤前的行号，也不能直接用。
# 两张表都是同一批订单的置换，用 (下单时间, 总运送周期) 双键排序可无损重排。
ORDER_KEY, ACTUAL_COL = "下单时间", "总运送周期"
pred_sorted = pred.sort_values([ORDER_KEY, ACTUAL_COL], kind="stable")
oof_sorted = oof_raw.rename(columns={"order_time": "下单时间", "total_delivery_time": "总运送周期"})
oof_sorted[ORDER_KEY] = pd.to_datetime(oof_sorted[ORDER_KEY], format="mixed")
oof_sorted = oof_sorted.sort_values([ORDER_KEY, ACTUAL_COL], kind="stable").reset_index(drop=True)

# 校验：两表排序后的键与标签必须逐行完全相同
if not (pred_sorted[ORDER_KEY].to_numpy() == oof_sorted[ORDER_KEY].to_numpy()).all():
    raise ValueError("折外预测表对齐失败：下单时间未能逐行匹配，请检查双键排序。")
if not np.isclose(pred_sorted[ACTUAL_COL].to_numpy(dtype=float),
                  oof_sorted[ACTUAL_COL].to_numpy(dtype=float)).all():
    raise ValueError("折外预测表对齐失败：真实总运送周期未能逐行匹配，请检查双键排序。")
print("折外预测表对齐校验通过：下单时间与真实周期逐行一致")

#%% 5. 折外预测值取整到 00:00（与 lgb_val.py 完全相同的公式）
# 折外表存的是未取整的连续天数，必须按部署口径换算后才能与全量表比较。
# 预计送达时间必须是某日 00:00:00：先加到下单时间，再向后取到下一个整日，
# 然后用该整日期反算天数差，得到与原算法「预期总周期」同口径的修正值。
order_time = pred_sorted[ORDER_KEY].reset_index(drop=True)
OOF_COLS = {}
for model_name in MODEL_NAMES:
    src_col = f"{model_name}_oof_预期总周期"
    if src_col not in oof_sorted.columns:
        print(f"折外表缺少列，已跳过：{src_col}")
        continue
    raw_eta = order_time + pd.to_timedelta(pd.Series(oof_sorted[src_col].to_numpy(dtype=float)), unit="D")
    eta_date = raw_eta.dt.normalize() + pd.Timedelta(days=1)
    OOF_COLS[model_name] = ((eta_date - order_time).dt.total_seconds() / 86400).round(2).to_numpy()

# 折外有效行：四个模型都有折外预测的订单
oof_valid = oof_sorted[[f"{m}_oof_预期总周期" for m in OOF_COLS]].notna().all(axis=1).to_numpy()
print(f"折外有效订单：{int(oof_valid.sum())} 行 | "
      f"时间范围：{order_time[oof_valid].min()} ~ {order_time[oof_valid].max()}")

#%% 6. 构造三套口径的预测结果：全量 / 时间外测试期 / 折外
# 全量口径含样本内乐观（模型见过这些订单的标签），时间外测试期与折外才是样本外。
# 折外模拟的是「持续增量训练后预测新单」，最接近真实部署。
actual_all = pred[ACTUAL_COL].to_numpy(dtype=float)
base_all = pred["预期总周期"].to_numpy(dtype=float)
full_models = {m: pred[f"{m}预期总周期"].to_numpy(dtype=float) for m in MODEL_NAMES}

# 时间外测试期：与 lgb_model.py 相同的切分点（order_time 后 test_size 比例）
cutoff_time = pd.to_datetime(pred["下单时间"]).quantile(1 - CONFIG["test_size"])
test_mask = (pd.to_datetime(pred["下单时间"]) >= cutoff_time).to_numpy()

# 折外：把折外预测值挂回原始行位置（排序位置 -> 原始行号）
orig_pos = pred_sorted.index.to_numpy()
oof_rows = orig_pos[oof_valid]
oof_models = {}
for m in OOF_COLS:
    arr = np.full(len(pred), np.nan)
    arr[oof_rows] = OOF_COLS[m][oof_valid]
    oof_models[m] = arr

# 三套口径共用同一批基准：模型来源不同，但真实值、原算法承诺、样本范围口径一致
SCOPE_MODEL_SRC = {"全量": full_models, "时间外测试期": full_models, "折外": oof_models}
SCOPE_MASK = {"全量": np.ones(len(pred), dtype=bool), "时间外测试期": test_mask}
# 折外口径的行范围单独标记
oof_scope_mask = np.zeros(len(pred), dtype=bool)
oof_scope_mask[oof_rows] = True
SCOPE_MASK["折外"] = oof_scope_mask

#%% 7. 评估与绘图函数
PCT_COLS = ["early_delivery_rate", "delay_rate", "exact_rate", "severe_overtime_rate"]


def build_metrics(actual, model_preds, base):
    """计算一组订单上各模型的预测精度与延迟风险（比例保存为 0~1）。"""
    rows = []
    for name, pred_days in [("Original_Algorithm", base)] + list(model_preds.items()):
        valid = np.isfinite(actual) & np.isfinite(pred_days) & (actual > 0) & (pred_days > 0)
        if valid.sum() == 0:
            continue
        a, p = actual[valid], pred_days[valid]
        error = p - a
        rows.append({
            "model": name, "sample_count": int(valid.sum()),
            "MAE_days": mean_absolute_error(a, p),
            "RMSE_days": np.sqrt(mean_squared_error(a, p)),
            "pred_mean_days": np.mean(p), "actual_mean_days": np.mean(a),
            "mean_error_days": np.mean(error),
            "early_delivery_rate": np.mean(error > 0), "delay_rate": np.mean(error < 0),
            "exact_rate": np.mean(np.isclose(p, a, rtol=1e-9, atol=1e-9)),
            "severe_overtime_rate": np.mean(error < -SEVERE_UNDER_DAYS),
        })
    return pd.DataFrame(rows)


def scope_frame(scope_name):
    """取出某个口径下的 (真实值, 原算法, 各模型预测)。"""
    mask = SCOPE_MASK[scope_name]
    models = SCOPE_MODEL_SRC[scope_name]
    return (actual_all[mask], {m: models[m][mask] for m in models}, base_all[mask])


#%% 8. 三套口径的评估：全量 / 时间外测试期 / 折外
scope_metrics = {}
for scope_name in ["全量", "时间外测试期", "折外"]:
    a, mp, b = scope_frame(scope_name)
    scope_metrics[scope_name] = build_metrics(a, mp, b)

# 汇总表：直接给出各口径下的延迟率与 MAE
summary_rows = []
for scope_name in ["全量", "时间外测试期", "折外"]:
    m = scope_metrics[scope_name].set_index("model")
    row = {"口径": scope_name, "样本数": int(m["sample_count"].iloc[0]),
           "原算法延迟率": m.loc["Original_Algorithm", "delay_rate"]}
    for model_name in MODEL_NAMES:
        if model_name in m.index:
            row[f"{model_name}_延迟率"] = m.loc[model_name, "delay_rate"]
            row[f"{model_name}_MAE"] = m.loc[model_name, "MAE_days"]
            row[f"{model_name}_平均承诺"] = m.loc[model_name, "pred_mean_days"]
    row["真实均值"] = m["actual_mean_days"].iloc[0]
    summary_rows.append(row)

scope_summary = pd.DataFrame(summary_rows)
scope_summary.to_csv(OUTPUT_DIR / "ooc_scope_summary.csv", index=False, encoding="utf-8-sig")

print("\n========== 三套口径的延迟率对照 ==========")
print("全量口径含样本内乐观；折外口径模拟「持续增量训练后预测新单」，最接近真实部署。")
show = scope_summary.copy()
for c in [c for c in show.columns if "延迟率" in c]:
    show[c] = (show[c] * 100).round(2)
print(show.round(3).to_string(index=False))

#%% 9. 折外口径的完整评估输出：指标、分布与图
oof_metrics = scope_metrics["折外"].sort_values("MAE_days").reset_index(drop=True)
oof_metrics.to_csv(OUTPUT_DIR / "ooc_timeout_model_comparison.csv", index=False, encoding="utf-8-sig")


def display_name(model_name):
    """图例与表头里使用的显示名。"""
    return "原算法" if model_name == "Original_Algorithm" else model_name


oof_result = oof_metrics.copy()
oof_result[PCT_COLS] *= 100
print("\n========== 折外订单：预期总周期评估 ==========")
print("提前送达率：预期总周期 > 真实总运送周期的订单占比。")
print("延迟率：预期总周期 < 真实总运送周期的订单占比。")
print(f"严重超时率：真实比预期多超过 {SEVERE_UNDER_DAYS:.1f} 天的订单占比。")
print("比例单位为 %；时长和误差单位为天。")
print(oof_result.round(3).to_string(index=False))

risk = oof_metrics[[
    "model", "sample_count", "MAE_days", "RMSE_days", "pred_mean_days", "actual_mean_days",
    "mean_error_days", "early_delivery_rate", "delay_rate", "exact_rate", "severe_overtime_rate",
]].copy()
risk[PCT_COLS] *= 100
risk.to_csv(OUTPUT_DIR / "ooc_mean_duration_and_risk_comparison.csv", index=False, encoding="utf-8-sig")

# 分布统计
dist_rows = []
a, mp, b = scope_frame("折外")
dist_rows.append({"数据来源": "折外真实总运送周期", "样本数": len(a), "均值": np.mean(a),
                  "标准差": np.std(a), "最小值": np.min(a), "P25": np.quantile(a, .25),
                  "中位数": np.median(a), "P75": np.quantile(a, .75), "P90": np.quantile(a, .90),
                  "P95": np.quantile(a, .95), "最大值": np.max(a)})
for name, vals in [("原算法", b)] + [(m, mp[m]) for m in MODEL_NAMES]:
    v = vals[np.isfinite(vals) & (vals > 0)]
    if len(v) == 0:
        continue
    dist_rows.append({"数据来源": f"{name}预期总周期", "样本数": len(v), "均值": np.mean(v),
                      "标准差": np.std(v), "最小值": np.min(v), "P25": np.quantile(v, .25),
                      "中位数": np.median(v), "P75": np.quantile(v, .75), "P90": np.quantile(v, .90),
                      "P95": np.quantile(v, .95), "最大值": np.max(v)})
pd.DataFrame(dist_rows).to_csv(OUTPUT_DIR / "ooc_delivery_time_distribution.csv",
                               index=False, encoding="utf-8-sig")

#%% 10. 折外订单的三张图
a, mp, b = scope_frame("折外")
valid = np.isfinite(a) & (a > 0)

plt.figure(figsize=CONFIG["figure_size"])
sns.kdeplot(a[valid], label="真实总运送周期", linewidth=2.8)
sns.kdeplot(b[valid], label="原算法", linewidth=2)
for m in MODEL_NAMES:
    v = mp[m][np.isfinite(mp[m]) & (mp[m] > 0)]
    if len(v) > 1:
        sns.kdeplot(v, label=m, linewidth=2)
plt.title("折外订单：原算法、各模型预期总周期与真实周期分布对比")
plt.xlabel("配送周期（天）")
plt.ylabel("概率密度")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(ASSET_DIR / "ooc_delivery_time_distribution.svg", dpi=CONFIG["figure_dpi"], bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=CONFIG["figure_size"])
for name, vals in [("Original_Algorithm", b)] + [(m, mp[m]) for m in MODEL_NAMES]:
    ok = np.isfinite(a) & (a > 0) & np.isfinite(vals) & (vals > 0)
    err = vals[ok] - a[ok]
    if len(err) > 1:
        sns.kdeplot(err, label=display_name(name), linewidth=2)
plt.axvline(0, color="red", linestyle="--", linewidth=1.8, label="零误差线")
plt.title("折外订单：原算法与各模型的预期周期误差分布")
plt.xlabel("预期总周期 - 真实总运送周期（天）")
plt.ylabel("概率密度")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(ASSET_DIR / "ooc_delivery_time_error_distribution.svg",
            dpi=CONFIG["figure_dpi"], bbox_inches="tight")
plt.show()
plt.close()

names = [display_name(m) for m in oof_metrics["model"].tolist()]
rates = oof_metrics["delay_rate"].to_numpy() * 100
fig, ax = plt.subplots(figsize=(13, 6), dpi=CONFIG["figure_dpi"])
bars = ax.bar(names, rates, width=0.65)
ax.set_title("折外订单：不同算法的配送延迟率对比")
ax.set_xlabel("模型名称")
ax.set_ylabel("延迟率（%）")
ax.set_xticks(np.arange(len(names)))
ax.set_xticklabels(names, rotation=25, ha="right")
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9)
ax.set_ylim(0, max(rates) * 1.15 if len(rates) else 1)
fig.tight_layout()
fig.savefig(ASSET_DIR / "ooc_delivery_delay_rate_comparison.svg",
            dpi=CONFIG["figure_dpi"], bbox_inches="tight")
plt.show()
plt.close(fig)

#%% 11. 关键诊断：同一批折外订单上，最终模型 vs 折外模型
# 两者预测的是同一批订单。差异全部来自「模型见过多少数据」：
# 最终模型用全量数据训练（含这批订单本身），折外模型只用其之前的订单训练。
rows = []
for m in MODEL_NAMES:
    fin = full_models[m][oof_scope_mask]
    out = oof_models[m][oof_scope_mask]
    ok = np.isfinite(fin) & np.isfinite(out) & np.isfinite(a) & (a > 0) & (fin > 0) & (out > 0)
    f, o, aa = fin[ok], out[ok], a[ok]
    rows.append({
        "模型": m,
        "最终_平均承诺": np.mean(f), "最终_延迟率": np.mean(aa > f), "最终_MAE": np.abs(f - aa).mean(),
        "最终_承诺标准差": np.std(f),
        "折外_平均承诺": np.mean(o), "折外_延迟率": np.mean(aa > o), "折外_MAE": np.abs(o - aa).mean(),
        "折外_承诺标准差": np.std(o),
        "承诺均值差": np.mean(o) - np.mean(f),
        "MAE变化率": np.abs(o - aa).mean() / np.abs(f - aa).mean() - 1,
        "两模型预测相关系数": np.corrcoef(f, o)[0, 1],
    })
in_vs_out = pd.DataFrame(rows)
in_vs_out.to_csv(OUTPUT_DIR / "ooc_in_vs_out_comparison.csv", index=False, encoding="utf-8-sig")

print("\n========== 同一批折外订单：最终模型 vs 折外模型 ==========")
print("差异全部来自「模型见过多少数据」：最终模型见过全量，折外模型只见过它之前的订单。")
show = in_vs_out.copy()
for c in ["最终_延迟率", "折外_延迟率", "MAE变化率"]:
    show[c] = (show[c] * 100).round(2)
print(show.round(3).to_string(index=False))

#%% 12. 按月诊断：真实配送时长是否在漂移
month = pd.to_datetime(pred["下单时间"]).dt.to_period("M")
drift_rows = []
for m, idx in pred.groupby(month).groups.items():
    ix = np.asarray(idx)
    a_m = actual_all[ix]
    mask_oof = oof_scope_mask[ix]
    row = {"月份": str(m), "订单数": len(ix), "实际均值": np.mean(a_m),
           "实际P90": np.quantile(a_m, .90), "原算法承诺": np.mean(base_all[ix])}
    for name in MODEL_NAMES:
        row[f"{name}_最终承诺"] = np.mean(full_models[name][ix])
        o20 = oof_models[name][ix][mask_oof]
        row[f"{name}_折外承诺"] = np.mean(o20) if len(o20) else np.nan
        row[f"{name}_折外延迟率"] = np.mean(a_m[mask_oof] > o20) if len(o20) else np.nan
    drift_rows.append(row)

monthly = pd.DataFrame(drift_rows)
monthly.to_csv(OUTPUT_DIR / "ooc_monthly_drift.csv", index=False, encoding="utf-8-sig")
print("\n========== 按月：真实配送时长的漂移 ==========")
print(f"真实均值最高月 {monthly.loc[monthly['实际均值'].idxmax(), '月份']} "
      f"{monthly['实际均值'].max():.2f} 天 | 最低月 "
      f"{monthly.loc[monthly['实际均值'].idxmin(), '月份']} {monthly['实际均值'].min():.2f} 天")
print("每月折外延迟率见 output/ooc_monthly_drift.csv")

#%% 13. 极端离群值诊断：按实际时长分桶，看各模型的预测是否跟得上
bins = [0, 5, 10, 15, 20, 25, 30, 40, 60, np.inf]
labels = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-40", "40-60", "60+"]
# 折外承诺取自折外口径的 mp；同批订单在最终模型下的表现需另行按行号取全量表
bucket_rows = []
sel_model = "PseudoHuber_Under20x"
for label, (lo, hi) in zip(labels, zip(bins[:-1], bins[1:])):
    sel = (a >= lo) & (a < hi)
    if sel.sum() < 30:
        continue
    row = {"实际时长区间": label, "订单数": int(sel.sum()), "实际均值": np.mean(a[sel]),
           "原算法承诺": np.mean(b[sel])}
    for name in MODEL_NAMES:
        row[f"{name}_折外承诺"] = np.mean(mp[name][sel])
        row[f"{name}_折外延迟率"] = np.mean(mp[name][sel] < a[sel])
        # 同批订单在最终模型下的表现（用于对照）
        idx = np.where(oof_scope_mask)[0][sel]
        row[f"{name}_最终承诺"] = np.mean(full_models[name][idx])
        row[f"{name}_最终延迟率"] = np.mean(full_models[name][idx] < actual_all[idx])
    bucket_rows.append(row)

buckets = pd.DataFrame(bucket_rows)
buckets.to_csv(OUTPUT_DIR / "ooc_extreme_diagnosis.csv", index=False, encoding="utf-8-sig")
print("\n========== 按实际时长分桶：折外模型的预测是否跟得上长尾 ==========")
print(f"以 {sel_model} 为例（完整结果见 output/ooc_extreme_diagnosis.csv）：")
print(f"{'区间':8s}{'订单数':>7}{'实际均值':>9}{'最终承诺':>9}{'最终延迟率':>10}"
      f"{'折外承诺':>9}{'折外延迟率':>10}")
for _, r in buckets.iterrows():
    print(f"{r['实际时长区间']:8s}{int(r['订单数']):7d}{r['实际均值']:9.2f}"
          f"{r[f'{sel_model}_最终承诺']:9.2f}{r[f'{sel_model}_最终延迟率'] * 100:9.1f}%"
          f"{r[f'{sel_model}_折外承诺']:9.2f}{r[f'{sel_model}_折外延迟率'] * 100:9.1f}%")

# 分桶图：最终模型 vs 折外模型的承诺水平
fig, ax = plt.subplots(figsize=CONFIG["figure_size"], dpi=CONFIG["figure_dpi"])
x = np.arange(len(buckets))
ax.plot(x, buckets["实际均值"], "k--", marker="o", linewidth=2, label="实际时长（理想参照）")
ax.plot(x, buckets[f"{sel_model}_最终承诺"], marker="s", linewidth=2, label=f"{sel_model}·最终模型")
ax.plot(x, buckets[f"{sel_model}_折外承诺"], marker="^", linewidth=2, label=f"{sel_model}·折外模型")
ax.plot(x, buckets["原算法承诺"], ":", marker="v", linewidth=2, color="gray", label="原算法承诺")
ax.set_xticks(x)
ax.set_xticklabels(buckets["实际时长区间"])
ax.set_xlabel("实际总运送周期区间（天）")
ax.set_ylabel("平均承诺天数")
ax.set_title("各实际时长区间下的平均承诺：折外模型对长尾订单的识别能力")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(ASSET_DIR / "ooc_extreme_bucket.png", dpi=CONFIG["figure_dpi"], bbox_inches="tight")
plt.show()
plt.close(fig)

# 极端订单的口径说明
extreme = a > np.quantile(a, 0.99)
print(f"\n实际时长 P99 = {np.quantile(a, 0.99):.2f} 天；超 P99 的订单 {int(extreme.sum())} 单"
      f"（占 {extreme.mean() * 100:.2f}%），最长 {a.max():.1f} 天")
print(f"这 {int(extreme.sum())} 单在 {sel_model} 折外口径下的延迟率："
      f"{np.mean(mp[sel_model][extreme] < a[extreme]) * 100:.2f}%")

#%% 14. 极端离群值的惩罚行为：非对称 Pseudo-Huber 的梯度
# 对数空间残差 x = log(y) - log(ŷ)；Pseudo-Huber 的梯度为 x / sqrt(1 + (x/delta)^2)，
# 当 |x| 远大于 delta 时梯度饱和到 ±delta，即在损失层面「封顶」了极端值的影响。
print("\n========== 非对称 Pseudo-Huber 对极端残差的惩罚行为 ==========")
print(f"delta = {DELTA}（作用于对数空间），低估惩罚倍率 = {UNDER_WEIGHTS}，高估倍率 = {OVER_WEIGHT}")
print("梯度 = 倍率 × x / sqrt(1 + (x/delta)^2)，倍率越大惩罚越强，但|x|>>delta 时梯度饱和。")
print(f"{'对数残差x':>10}{'相对误差':>10}{'L1梯度':>9}{'L2梯度':>9}"
      + "".join([f"{'5x' if w == 5 else str(int(w)) + 'x':>9}" for w in UNDER_WEIGHTS]))


def huber_grad(x, delta):
    return x / np.sqrt(1 + (x / delta) ** 2)


grad_rows = []
for x in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5]:
    row = {"对数残差x": x, "相对误差": np.exp(abs(x)) - 1,
           "L1梯度": 1.0, "L2梯度": abs(x)}
    for w in UNDER_WEIGHTS:
        row[f"低估{w:g}x梯度"] = w * abs(huber_grad(abs(x), DELTA))
    row[f"高估{OVER_WEIGHT:g}x梯度"] = OVER_WEIGHT * abs(huber_grad(abs(x), DELTA))
    grad_rows.append(row)
    line = f"{x:10.2f}{np.exp(abs(x)) - 1:9.1%}{1.0:9.2f}{abs(x):9.2f}"
    line += "".join([f"{w * abs(huber_grad(abs(x), DELTA)):9.2f}" for w in UNDER_WEIGHTS])
    print(line)
pd.DataFrame(grad_rows).to_csv(OUTPUT_DIR / "ooc_loss_gradient.csv", index=False, encoding="utf-8-sig")
print("结论：残差超过 delta 后梯度饱和，极端离群值在损失中的权重被封顶，不会被个别订单主导；")
print("      对「低估」方向再乘 20~40 倍，才是让模型主动变保守的原因。")

#%% 15. 导出折外预测表，供 GMV_analysis 直接复用
# row_pos 是该行在 LightGBM预测数据.csv 中的行号，下游按行号直接对齐即可，
# 既不必再处理行序，也不必再补取整（本文件存的就是取整后的部署口径）。
export = pd.DataFrame({
    "row_pos": orig_pos[oof_valid],
    "下单时间": pred_sorted[ORDER_KEY].to_numpy()[oof_valid],
    "总运送周期": pred_sorted[ACTUAL_COL].to_numpy()[oof_valid],
})
for m in OOF_COLS:
    export[f"{m}_ooc_预期总周期"] = OOF_COLS[m][oof_valid]
export = export.sort_values("row_pos").reset_index(drop=True)
export_path = OUTPUT_DIR / "ooc_predictions.csv"
export.to_csv(export_path, index=False, encoding="utf-8-sig")
print(f"\n折外预测值已导出：{export_path}（{len(export)} 行，含取整后的各模型折外承诺）")
print(f"表格保存目录：{OUTPUT_DIR.resolve()}")
print(f"图表保存目录：{ASSET_DIR.resolve()}")
