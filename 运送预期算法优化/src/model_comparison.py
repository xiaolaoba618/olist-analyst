# %% 1. 脚本说明
"""
LightGBM 与 Transformer+VCN 同倍率对比脚本

职责：
  1) 读取两个模型各自训练脚本 / val 脚本 / ooc 脚本产出的指标表与预测明细（不重新训练）；
  2) 按相同非对称惩罚倍率（L1 / 5x / 20x / 40x）并排对比 MAE、延迟率、承诺水平等；
  3) 覆盖全量 / 时间外测试期 / 折外三套口径，折外明细按 row_pos 内连接逐单对齐；
  4) 输出对比表 CSV 与对比图 SVG（指标柱状、周期分布、误差分布、倍率趋势、三口径对照）。

口径约定：
  - 全部数据为 2017-09-01 及以后的 72,907 条订单；验证集为时间后 20% 的 14,582 条；
  - 折外为时间序列 5 折前向链，LGB 58,326 条、TCNN 58,325 条，按 row_pos 取交集 58,325 条逐单对齐；
  - 「验证集 / 折外」表均为取整到 00:00 后的部署口径，与 分析.md 中 LightGBM 小节一致；
  - 严重超时 = 真实比承诺多超过 config.severe_under_days 天；
  - 比例统一按 0~1 保存在 CSV，图中再乘 100 显示为百分数。

依赖：先运行 Transformer_CNN_model.py -> Transformer_CNN_model_val.py -> Transformer_CNN_model_ooc.py（本脚本只读产出，不改模型）。
运行：用项目 .venv 的 python 执行本脚本。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 规避 Windows 上 torch/MKL 双 OpenMP 运行库冲突

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
OUTPUT_DIR = BASE_DIR / "output"
ASSET_DIR = BASE_DIR / "assets"

from config import CONFIG

# %% 2. 全局设置
LEVELS = ["L1", "5x", "20x", "40x"]
PCT_COLS = ["early_delivery_rate", "delay_rate", "exact_rate", "severe_overtime_rate"]
MODEL_COLS = [
    "MAE_days", "RMSE_days", "pred_mean_days", "actual_mean_days", "mean_error_days",
    "early_delivery_rate", "delay_rate", "severe_overtime_rate",
]

# 模型原名 -> 倍率标签
LGB_MAP = {
    "L1": "L1", "PseudoHuber_Under5x": "5x",
    "PseudoHuber_Under20x": "20x", "PseudoHuber_Under40x": "40x",
}
TCNN_MAP = {
    "TransCNN_L1": "L1", "TransCNN_PseudoHuber_Under5x": "5x",
    "TransCNN_PseudoHuber_Under20x": "20x", "TransCNN_PseudoHuber_Under40x": "40x",
}

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
FIG_DPI = CONFIG["figure_dpi"]
FAMILY_COLOR = {"LightGBM": "#2f6fbf", "TransformerCNN": "#e07b28"}


# %% 3. 数据读取工具
def _to_fraction(df):
    """两侧指标表有百分数(0~100)与小数(0~1)两种落盘口径，统一成 0~1。"""
    out = df.copy()
    scale = 100.0 if out["delay_rate"].max(skipna=True) > 1.5 else 1.0
    for c in PCT_COLS:
        if c in out.columns:
            out[c] = out[c] / scale
    return out


def load_metrics(path, model_map):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = _to_fraction(df)
    df["level"] = df["model"].map(model_map)
    df = df[df["level"].notna()].set_index("level")
    missing = set(LEVELS) - set(df.index)
    if missing:
        raise ValueError(f"{path.name} 缺少倍率：{missing}")
    return df.loc[LEVELS]


def load_test_predictions(path, model_map):
    raw = pd.read_csv(path, encoding="utf-8-sig")
    frames = []
    for model_name, level in model_map.items():
        sub = raw[raw["model"] == model_name]
        frames.append(pd.DataFrame({
            "family": "TransformerCNN" if "TransCNN" in model_name else "LightGBM",
            "level": level,
            "actual_days": pd.to_numeric(sub["actual_days"], errors="coerce").to_numpy(),
            "pred_days": pd.to_numeric(sub["pred_days"], errors="coerce").to_numpy(),
        }))
    out = pd.concat(frames, ignore_index=True)
    return out[np.isfinite(out["actual_days"]) & np.isfinite(out["pred_days"])]


def load_full_predictions(path, model_map):
    wide = pd.read_csv(path, encoding="utf-8-sig",
                       usecols=["总运送周期"] + [m + "预期总周期" for m in model_map])
    actual = pd.to_numeric(wide["总运送周期"], errors="coerce")
    frames = []
    for model_name, level in model_map.items():
        pred = pd.to_numeric(wide[model_name + "预期总周期"], errors="coerce")
        ok = actual.notna() & pred.notna() & (actual > 0) & (pred > 0)
        frames.append(pd.DataFrame({
            "family": "TransformerCNN" if "TransCNN" in model_name else "LightGBM",
            "level": level,
            "actual_days": actual[ok].to_numpy(),
            "pred_days": pred[ok].to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True)


def load_ooc_detail(lgb_path, tcnn_path, lgb_map, tcnn_map):
    """折外明细按 row_pos 内连接逐单对齐，返回两模型合一的 detail。"""
    lgb = pd.read_csv(lgb_path, encoding="utf-8-sig")
    tcnn = pd.read_csv(tcnn_path, encoding="utf-8-sig")
    merged = lgb.merge(tcnn, on="row_pos", how="inner", suffixes=("_lgb", "_tcnn"))
    actual = pd.to_numeric(merged["总运送周期"], errors="coerce")
    frames = []
    for model_name, level in lgb_map.items():
        pred = pd.to_numeric(merged[model_name + "_ooc_预期总周期"], errors="coerce")
        ok = actual.notna() & pred.notna() & (actual > 0) & (pred > 0)
        frames.append(pd.DataFrame({
            "family": "LightGBM", "level": level,
            "actual_days": actual[ok].to_numpy(), "pred_days": pred[ok].to_numpy()}))
    for model_name, level in tcnn_map.items():
        pred = pd.to_numeric(merged[model_name + "_ooc_预期总周期"], errors="coerce")
        ok = actual.notna() & pred.notna() & (actual > 0) & (pred > 0)
        frames.append(pd.DataFrame({
            "family": "TransformerCNN", "level": level,
            "actual_days": actual[ok].to_numpy(), "pred_days": pred[ok].to_numpy()}))
    print(f"折外明细按 row_pos 对齐：{len(merged)} 条（LGB 独有 1 条已剔除）")
    return pd.concat(frames, ignore_index=True)


def build_compare(lgb_metrics, tcnn_metrics, scope_name):
    rows = []
    for level in LEVELS:
        row = {"倍率": level}
        for col in MODEL_COLS:
            lv = float(lgb_metrics.loc[level, col])
            tv = float(tcnn_metrics.loc[level, col])
            row[f"LGB_{col}"] = lv
            row[f"TCNN_{col}"] = tv
            row[f"差值_{col}"] = tv - lv
        rows.append(row)
    original = {}
    for col in MODEL_COLS:
        vals = []
        for df in (lgb_metrics, tcnn_metrics):
            if "Original_Algorithm" in df.index:
                vals.append(float(df.loc["Original_Algorithm", col]))
        original[f"原算法_{col}"] = float(np.mean(vals)) if vals else np.nan
    out = pd.DataFrame(rows)
    out.attrs["scope"] = scope_name
    out.attrs["original"] = original
    return out


# %% 4. 读取四张指标表与两套预测明细
print("读取 LGB / TransformerCNN 产出表 ...")
lgb_test_m = load_metrics(OUTPUT_DIR / "val_test_mean_duration_and_risk_comparison.csv", LGB_MAP)
tcnn_test_m = load_metrics(OUTPUT_DIR / "tcnn_val_test_mean_duration_and_risk_comparison.csv", TCNN_MAP)
lgb_full_m = load_metrics(OUTPUT_DIR / "val_mean_duration_and_risk_comparison.csv", LGB_MAP)
tcnn_full_m = load_metrics(OUTPUT_DIR / "tcnn_val_mean_duration_and_risk_comparison.csv", TCNN_MAP)
lgb_ooc_m = load_metrics(OUTPUT_DIR / "ooc_mean_duration_and_risk_comparison.csv", LGB_MAP)
tcnn_ooc_m = load_metrics(OUTPUT_DIR / "tcnn_ooc_mean_duration_and_risk_comparison.csv", TCNN_MAP)

lgb_test_p = load_test_predictions(OUTPUT_DIR / "timeout_model_predictions.csv", LGB_MAP)
tcnn_test_p = load_test_predictions(OUTPUT_DIR / "tcnn_model_predictions.csv", TCNN_MAP)
lgb_full_p = load_full_predictions(OUTPUT_DIR / "LightGBM预测数据.csv", LGB_MAP)
tcnn_full_p = load_full_predictions(OUTPUT_DIR / "TransformerCNN预测数据.csv", TCNN_MAP)
test_detail = pd.concat([lgb_test_p, tcnn_test_p], ignore_index=True)
full_detail = pd.concat([lgb_full_p, tcnn_full_p], ignore_index=True)
ooc_detail = load_ooc_detail(
    OUTPUT_DIR / "ooc_predictions.csv", OUTPUT_DIR / "tcnn_ooc_predictions.csv",
    LGB_MAP, TCNN_MAP)

test_compare = build_compare(lgb_test_m, tcnn_test_m, "时间外测试期")
full_compare = build_compare(lgb_full_m, tcnn_full_m, "全量订单")
ooc_compare = build_compare(lgb_ooc_m, tcnn_ooc_m, "折外")

# %% 5. 输出对比表
def save_compare(compare, file_name):
    out = compare.copy()
    for c in PCT_COLS:
        for fam in ("LGB", "TCNN", "差值"):
            key = f"{fam}_{c}"
            if key in out.columns:
                out[key] = out[key] * 100
    out.round(4).to_csv(OUTPUT_DIR / file_name, index=False, encoding="utf-8-sig")


save_compare(test_compare, "model_comparison_test.csv")
save_compare(full_compare, "model_comparison_full.csv")
save_compare(ooc_compare, "model_comparison_ooc.csv")

# 三口径对照表：每口径每倍率一行，便于直接看全量乐观、测试期偏快、折外回落的落差
scope_rows = []
for scope, cmp, n in [("全量（含样本内）", full_compare, 72907),
                     ("时间外测试期", test_compare, 14582),
                     ("折外（5 折前向链）", ooc_compare, len(ooc_detail) // len(LEVELS) // 2)]:
    for level in LEVELS:
        r = cmp[cmp["倍率"] == level].iloc[0]
        scope_rows.append({
            "口径": scope, "倍率": level, "样本数": n,
            "LGB_延迟率%": r["LGB_delay_rate"] * 100,
            "TCNN_延迟率%": r["TCNN_delay_rate"] * 100,
            "延迟率差(pp)": r["差值_delay_rate"] * 100,
            "LGB_MAE": r["LGB_MAE_days"], "TCNN_MAE": r["TCNN_MAE_days"],
            "MAE差(TCNN-LGB)": r["差值_MAE_days"],
            "LGB_承诺均值": r["LGB_pred_mean_days"],
            "TCNN_承诺均值": r["TCNN_pred_mean_days"],
        })
pd.DataFrame(scope_rows).round(4).to_csv(
    OUTPUT_DIR / "model_comparison_scope.csv", index=False, encoding="utf-8-sig")

summary_rows = []
for scope, cmp, detail in [("时间外测试期", test_compare, test_detail),
                           ("全量订单", full_compare, full_detail),
                           ("折外", ooc_compare, ooc_detail)]:
    for level in LEVELS:
        r = cmp[cmp["倍率"] == level].iloc[0]
        summary_rows.append({
            "口径": scope, "倍率": level,
            "LGB_MAE": r["LGB_MAE_days"], "TCNN_MAE": r["TCNN_MAE_days"],
            "MAE差(TCNN-LGB)": r["差值_MAE_days"],
            "LGB_延迟率%": r["LGB_delay_rate"] * 100, "TCNN_延迟率%": r["TCNN_delay_rate"] * 100,
            "延迟率差(pp)": r["差值_delay_rate"] * 100,
            "LGB_承诺均值": r["LGB_pred_mean_days"], "TCNN_承诺均值": r["TCNN_pred_mean_days"],
            "承诺差(天)": r["差值_pred_mean_days"],
            "样本数": int((detail["level"] == level).sum() / 2),
        })
pd.DataFrame(summary_rows).round(4).to_csv(
    OUTPUT_DIR / "model_comparison_summary.csv", index=False, encoding="utf-8-sig")

print(f"\n===== 时间外测试期（14,582 条）=====")
print(test_compare[["倍率", "LGB_MAE_days", "TCNN_MAE_days", "LGB_delay_rate",
                    "TCNN_delay_rate", "LGB_pred_mean_days", "TCNN_pred_mean_days"]]
      .assign(LGB_delay_rate=lambda d: d["LGB_delay_rate"] * 100,
              TCNN_delay_rate=lambda d: d["TCNN_delay_rate"] * 100)
      .round(3).to_string(index=False))
print(f"\n===== 全量订单（72,907 条）=====")
print(full_compare[["倍率", "LGB_MAE_days", "TCNN_MAE_days", "LGB_delay_rate",
                    "TCNN_delay_rate", "LGB_pred_mean_days", "TCNN_pred_mean_days"]]
      .assign(LGB_delay_rate=lambda d: d["LGB_delay_rate"] * 100,
              TCNN_delay_rate=lambda d: d["TCNN_delay_rate"] * 100)
      .round(3).to_string(index=False))
print(f"\n===== 折外订单（{len(ooc_detail)//len(LEVELS)//2:,} 条，row_pos 对齐）=====")
print(ooc_compare[["倍率", "LGB_MAE_days", "TCNN_MAE_days", "LGB_delay_rate",
                   "TCNN_delay_rate", "LGB_pred_mean_days", "TCNN_pred_mean_days"]]
      .assign(LGB_delay_rate=lambda d: d["LGB_delay_rate"] * 100,
              TCNN_delay_rate=lambda d: d["TCNN_delay_rate"] * 100)
      .round(3).to_string(index=False))


# %% 6. 绘图工具
def savefig(name):
    plt.tight_layout()
    plt.savefig(ASSET_DIR / name, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def plot_metric_bars(compare, scope, figure_name):
    """同倍率下两模型 MAE 与延迟率并排柱状图。"""
    x = np.arange(len(LEVELS))
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    settings = [
        ("MAE_days", "MAE（天，越低越好）", 1),
        ("delay_rate", "延迟率（%，越低越好）", 100),
    ]
    for ax, (col, ylabel, scale) in zip(axes, settings):
        lv = compare[f"LGB_{col}"].to_numpy() * scale
        tv = compare[f"TCNN_{col}"].to_numpy() * scale
        b1 = ax.bar(x - w / 2, lv, w, label="LightGBM", color=FAMILY_COLOR["LightGBM"])
        b2 = ax.bar(x + w / 2, tv, w, label="TransformerCNN", color=FAMILY_COLOR["TransformerCNN"])
        for bars in (b1, b2):
            for bar in bars:
                ax.annotate(f"{bar.get_height():.2f}",
                            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(LEVELS)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{scope}：{ylabel.split('（')[0]}对比")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
    plt.suptitle(f"{scope}：相同惩罚倍率下 LightGBM 与 TransformerCNN 核心指标", y=1.03)
    savefig(figure_name)


def _kde(ax, values, label, color, linewidth=2.0, linestyle="-"):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return
    try:
        from scipy.stats import gaussian_kde
        grid = np.linspace(values.min(), values.max(), 220)
        ax.plot(grid, gaussian_kde(values)(grid), label=label,
                color=color, linewidth=linewidth, linestyle=linestyle)
    except ImportError:
        ax.hist(values, bins=70, density=True, histtype="step",
                label=label, color=color, linewidth=linewidth, linestyle=linestyle)


def plot_facet_distributions(detail, scope, figure_name, kind):
    """2x2 分面：每个倍率下两模型周期分布或误差分布的直接对照。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    is_error = kind == "error"
    for ax, level in zip(axes.flat, LEVELS):
        for family in ("LightGBM", "TransformerCNN"):
            sub = detail[(detail["level"] == level) & (detail["family"] == family)]
            values = (sub["pred_days"] - sub["actual_days"]).to_numpy() if is_error \
                else sub["pred_days"].to_numpy()
            _kde(ax, values, family, FAMILY_COLOR[family])
        if not is_error:
            actual = detail[detail["level"] == level]["actual_days"].unique()
            _kde(ax, detail[detail["level"] == level]["actual_days"].to_numpy(),
                 "真实总运送周期", "#777777", linewidth=2.4, linestyle="--")
        if is_error:
            ax.axvline(0, color="red", linestyle="--", linewidth=1.4, label="零误差线")
        ax.set_title(f"倍率 {level}")
        ax.set_xlabel("预期减去真实（天）" if is_error else "配送周期（天）")
        ax.set_ylabel("概率密度")
        ax.grid(alpha=0.22)
        if level == "L1":
            ax.legend(fontsize=8)
    title = (f"{scope}：同倍率下两模型预测误差分布" if is_error
             else f"{scope}：同倍率下两模型承诺周期与真实周期分布")
    plt.suptitle(title, y=1.01, fontsize=14)
    savefig(figure_name)


def plot_trends(scopes, figure_name):
    """倍率从 L1 到 40x，每个口径一组两模型走势（全量/测试期/折外）。"""
    x = np.arange(len(LEVELS))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    styles = ["-", "--", ":"]
    for ax, (col, ylabel, scale) in zip(axes, [
        ("MAE_days", "MAE（天）", 1),
        ("delay_rate", "延迟率（%）", 100),
    ]):
        for (scope, cmp), style in zip(scopes, styles):
            for family, prefix in [("LightGBM", "LGB"), ("TransformerCNN", "TCNN")]:
                y = cmp[f"{prefix}_{col}"].to_numpy() * scale
                ax.plot(x, y, marker="o", linewidth=2.0, color=FAMILY_COLOR[family],
                        linestyle=style, label=f"{family}·{scope}")
        ax.set_xticks(x)
        ax.set_xticklabels(LEVELS)
        ax.set_xlabel("非对称惩罚倍率")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5)
    plt.suptitle("惩罚倍率变化时两模型的精度与延迟风险走势（三口径）", y=1.03)
    savefig(figure_name)


def plot_scope_delay(compares, figure_name):
    """按倍率分面，横轴三口径，直观看全量→测试期→折外的延迟率落差。"""
    scope_names = ["全量", "测试期", "折外"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    for ax, level in zip(axes.flat, LEVELS):
        lgb = [c[c["倍率"] == level]["LGB_delay_rate"].iloc[0] * 100 for c in compares]
        tcnn = [c[c["倍率"] == level]["TCNN_delay_rate"].iloc[0] * 100 for c in compares]
        x = np.arange(len(scope_names))
        ax.plot(x, lgb, marker="o", linewidth=2.2, color=FAMILY_COLOR["LightGBM"], label="LightGBM")
        ax.plot(x, tcnn, marker="s", linewidth=2.2, color=FAMILY_COLOR["TransformerCNN"], label="TransformerCNN")
        for i, (a, b) in enumerate(zip(lgb, tcnn)):
            ax.annotate(f"{a:.1f}", (i, a), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8, color=FAMILY_COLOR["LightGBM"])
            ax.annotate(f"{b:.1f}", (i, b), textcoords="offset points", xytext=(0, -13),
                        ha="center", fontsize=8, color=FAMILY_COLOR["TransformerCNN"])
        ax.set_xticks(x)
        ax.set_xticklabels(scope_names)
        ax.set_title(f"倍率 {level}")
        ax.set_ylabel("延迟率（%）")
        ax.grid(alpha=0.25)
        if level == "L1":
            ax.legend(fontsize=8)
    plt.suptitle("三口径下两模型延迟率对比：全量乐观、测试期偏快、折外回落", y=1.01, fontsize=14)
    savefig(figure_name)


# %% 7. 出图
print("\n生成对比图 ...")
plot_metric_bars(test_compare, "时间外测试期", "model_comparison_test_metrics.svg")
plot_facet_distributions(test_detail, "时间外测试期", "model_comparison_test_distribution.svg", "cycle")
plot_facet_distributions(test_detail, "时间外测试期", "model_comparison_test_error.svg", "error")
plot_metric_bars(full_compare, "全量订单", "model_comparison_full_metrics.svg")
plot_facet_distributions(full_detail, "全量订单", "model_comparison_full_distribution.svg", "cycle")
plot_facet_distributions(full_detail, "全量订单", "model_comparison_full_error.svg", "error")
plot_metric_bars(ooc_compare, "折外订单", "model_comparison_ooc_metrics.svg")
plot_facet_distributions(ooc_detail, "折外订单", "model_comparison_ooc_distribution.svg", "cycle")
plot_facet_distributions(ooc_detail, "折外订单", "model_comparison_ooc_error.svg", "error")
plot_trends([("全量", full_compare), ("测试期", test_compare), ("折外", ooc_compare)],
            "model_comparison_trend.svg")
plot_scope_delay([full_compare, test_compare, ooc_compare], "model_comparison_scope_delay.svg")

print("完成。输出位置：")
print("  表格 output/model_comparison_{test,full,ooc,scope,summary}.csv")
print("  图片 assets/model_comparison_*.svg")
