from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # 无交互环境下出图（Windows 计划任务/IDE 后台均可跑）
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C

plt.rcParams["font.sans-serif"] = C.FONT_SANS
plt.rcParams["axes.unicode_minus"] = False


# =========================================================
# 1. 读取数据与对齐校验
# =========================================================
def load_inputs():
    """读入三张输入表，并把 GMV 口径与 ETA 预测对齐到同一份 frame。

    GMV分析.csv 与 LightGBM预测数据.csv 同为 72,907 行且行序一致，但本函数不做假设：
    逐行校验 5 个数值列 + 5 个标识列，任何一处不一致直接抛错。
    """
    gmv = pd.read_csv(C.GMV_CSV, encoding="utf-8-sig")
    eta = pd.read_csv(C.ETA_CSV, encoding="utf-8-sig")
    route = pd.read_csv(C.ROUTE_CSV, encoding="utf-8-sig")

    if len(gmv) != len(eta):
        raise ValueError(
            f"两张表行数不一致：GMV分析 {len(gmv)} 行 vs LightGBM预测数据 {len(eta)} 行，"
            "请先在运送预期算法优化中重跑 lgb_val.py 生成最新预测。"
        )

    for gcol, ecol in C.ALIGN_CHECK_NUMERIC:
        a = pd.to_numeric(gmv[gcol], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(eta[ecol], errors="coerce").to_numpy(dtype=float)
        bad = int((~np.isclose(a, b, atol=1e-6, equal_nan=True)).sum())
        if bad:
            raise ValueError(f"行对齐校验失败：{gcol} vs {ecol} 有 {bad} 行不一致。")
    for col in C.ALIGN_CHECK_OBJECT:
        bad = int((gmv[col].astype(str).to_numpy() != eta[col].astype(str).to_numpy()).sum())
        if bad:
            raise ValueError(f"行对齐校验失败：{col} 有 {bad} 行不一致。")

    _promise = pd.to_numeric(gmv[C.COL["promise"]], errors="coerce").to_numpy(dtype=float)
    _actual = pd.to_numeric(gmv[C.COL["actual"]], errors="coerce").to_numpy(dtype=float)
    frame = pd.DataFrame({
        "order_time": pd.to_datetime(eta[C.COL_ETA["order_time"]], errors="coerce"),
        "price": pd.to_numeric(gmv[C.COL["price"]], errors="coerce"),
        "promise_base": _promise,
        "actual": _actual,
        "delay_days_base": delay_days_of(_actual, _promise),
        "is_late_base": is_late(_actual, _promise),
        "is_logistics": gmv[C.COL["cause"]].eq(C.CAUSE_LOGISTICS).to_numpy(),
        "score": pd.to_numeric(gmv[C.COL["score"]], errors="coerce"),
        "seller_state": gmv[C.COL["seller_state"]].astype(str),
        "customer_state": gmv[C.COL["customer_state"]].astype(str),
    })
    # 反查：整天口径是否真的复刻了 GMV 表的送达状态（不复刻说明上游口径变了，必须停下来）
    label_late = gmv[C.COL["status"]].eq(C.STATUS_DELAYED).to_numpy()
    label_delay_days = pd.to_numeric(gmv[C.COL["delay_days"]], errors="coerce").to_numpy(dtype=float)
    same_late = (frame["is_late_base"].to_numpy() == label_late).all()
    same_days = np.isclose(frame["delay_days_base"].to_numpy(dtype=float), label_delay_days).all()
    if not (same_late and same_days):
        raise ValueError("延迟判定规则与 GMV分析.csv 的送达状态不一致，请检查上游数据口径。")
    print("  行对齐校验通过（数值/标识列逐行一致）；延迟判定采用「四舍五入到整天」规则，"
          "与 GMV 表送达状态 100% 吻合")

    # 线路级阶梯延时的推荐天数（③方案C 用）
    route_map = route.set_index([C.COL["seller_state"], C.COL["customer_state"]])["推荐延迟天数"].to_dict()
    frame["route_days"] = [
        float(route_map.get((s, c), 0.0)) for s, c in zip(frame["seller_state"], frame["customer_state"])
    ]

    # 新算法的 ETA（lgb_val.py 已按「向后取整到 00:00」规则回填，与客户实际看到的承诺一致）
    for m in C.MODEL_SOURCE_COLS:
        col = f"{m}{C.MODEL_ETA_SUFFIX}"
        if col not in eta.columns:
            raise ValueError(f"预测表缺少 {col}，请检查 config.MODEL_SOURCE_COLS 与 lgb_val.py 是否同步。")
        frame[f"eta_{m}"] = pd.to_numeric(eta[col], errors="coerce").to_numpy(dtype=float)

    frame = align_ooc(frame)
    return frame, route


def align_ooc(frame):
    """把折外预测直接按行号挂回 frame。

    折外预测由 运送预期算法优化/src/ooc_val.py 统一产出（output/ooc_predictions.csv），
    它已经完成了两件事：行序对齐（row_pos 列给出该行在 LightGBM预测数据.csv 中的行号）
    与向上取整到 00:00（存的就是部署口径的承诺天数）。
    因此这里直接按行号赋值，不重复做双键排序与取整换算。
    """
    for m in C.MODEL_SOURCE_COLS:
        frame[f"oof_{m}"] = np.nan

    if not C.OOC_CSV.exists():
        print(f"  未找到 {C.OOC_CSV.name}，跳过 OOF 折外口径。")
        return frame

    ooc = pd.read_csv(C.OOC_CSV, encoding="utf-8-sig")
    missing = [m for m in C.MODEL_SOURCE_COLS
               if f"{m}{C.OOC_ETA_SUFFIX}" not in ooc.columns]
    if missing or "row_pos" not in ooc.columns:
        print(f"  折外预测表缺少 {missing or 'row_pos'}，跳过 OOF 折外口径。")
        return frame

    pos = ooc["row_pos"].to_numpy(dtype=int)
    if pos.min() < 0 or pos.max() >= len(frame):
        raise ValueError("折外预测表的 row_pos 超出预测表范围，请重新运行 ooc_val.py。")
    for m in C.MODEL_SOURCE_COLS:
        frame.loc[pos, f"oof_{m}"] = pd.to_numeric(
            ooc[f"{m}{C.OOC_ETA_SUFFIX}"], errors="coerce").to_numpy(dtype=float)

    valid = frame[[f"oof_{m}" for m in C.MODEL_SOURCE_COLS]].notna().all(axis=1)
    print(f"  折外预测已挂回：{int(valid.sum())} 行可用"
          f"（{frame.loc[valid, 'order_time'].min().date()} ~ "
          f"{frame.loc[valid, 'order_time'].max().date()}）")
    return frame


# =========================================================
# 2. 核心口径函数
# =========================================================
def effective_days(delta_days, base_promise, t_ref, cap=None, gain_ratio=1.0):
    """把承诺时长的变化量折算成「相对基准时效的等效天数」。

    Δ>0（承诺变长）→ eff>0，计入转化流失；
    Δ<0（承诺变短）→ eff<0，计入转化收益，并可按 gain_ratio 打折
    （文献值只校准了加时方向，减时方向是否对称无依据，故留出保守化开关）。
    cap 用于把等效天数封顶，避免个别极端订单在复利模型里算出「转化翻倍」这种不可信的数。
    """
    base_safe = np.where(base_promise > 0, base_promise, np.nan)
    eff = delta_days * (t_ref / base_safe)
    eff = np.where(eff < 0, eff * gain_ratio, eff)
    if cap is not None:
        eff = np.clip(eff, -abs(cap), abs(cap))
    return np.nan_to_num(eff, nan=0.0, posinf=0.0, neginf=0.0)


def conversion_delta(price, eff, daily_rate=C.DAILY_LOSS_RATE):
    """逐单转化损益合计（元）。返回值 >0 表示 GMV 增加，<0 表示 GMV 流失。"""
    return float((price * ((1.0 - daily_rate) ** eff - 1.0)).sum())


def is_late(actual_days, promised_days):
    """统一的延迟判定：承诺时长与实际时长都四舍五入到整天后再比较。

    这是 GMV分析.csv 原本的规则（已验证：round(实际总运送周期) 与 round(预计总运送周期)
    比较的结果与「送达状态」列 100% 一致，延迟天数_天 也完全等于二者之差），
    因此整套账必须用同一把尺子——否则基准场景本身会凭空多出几百单「减少延迟」。

    注意这与运送预期算法优化项目内部的「连续口径（实际>承诺）」略有差异
    （约 0.4 个百分点），本模块统一采用 GMV 侧的整天口径。
    """
    return np.round(actual_days) > np.round(promised_days)


def delay_days_of(actual_days, promised_days):
    """相对某套承诺而言的超时天数（整数天，未超时记 0）。"""
    return np.maximum(np.round(actual_days) - np.round(promised_days), 0.0)


def build_scenario_eta(frame, scenario, use_oof=False):
    """构造场景承诺时长：ETA 来源 + 延时规则 → 每单客户看到的承诺天数。"""
    src = scenario["source"]
    if src == "原算法":
        eta = frame["promise_base"].to_numpy(dtype=float).copy()
    else:
        col = f"oof_{src}" if use_oof else f"eta_{src}"
        if col not in frame.columns:
            raise ValueError(f"frame 中缺少列 {col}")
        eta = frame[col].to_numpy(dtype=float).copy()

    rule = scenario["extend"]
    if rule == "none":
        pass
    elif rule == "fixed":
        add = np.full(len(frame), float(scenario.get("days", 2)))
        if scenario.get("target") == "logistics":
            # 方案B：只给「确实超时且归因为物流」的订单加时。
            # 注意这依赖事后结果（上帝视角），原报告把它当作订单级 ETA 的能力上限，
            # 真实可部署的是「按模型/线路预测风险提前加时」，见⑨组合场景。
            add = np.where(frame["is_late_base"].to_numpy() & frame["is_logistics"].to_numpy(), add, 0.0)
        eta = eta + add
    elif rule == "route":
        eta = eta + frame["route_days"].to_numpy(dtype=float)
    else:
        raise ValueError(f"未知延时规则：{rule}")
    return eta


def evaluate_scenario(frame, eta_new, *, t_ref, logistics_total, star1_delta,
                      daily_rate=C.DAILY_LOSS_RATE, cap=None, gain_ratio=None):
    """单点估计：算出一个场景的转化损益、履约改善与口碑收益。

    cap / gain_ratio 传 None 时取 config 的 MAX_EFF_DAYS / SHORTEN_GAIN_RATIO，
    这样改配置即刻生效，不必重启会话（敏感性分析要反复切换这两个值）。
    """
    cap = C.MAX_EFF_DAYS if cap is None else cap
    gain_ratio = C.SHORTEN_GAIN_RATIO if gain_ratio is None else gain_ratio
    price = frame["price"].to_numpy(dtype=float)
    actual = frame["actual"].to_numpy(dtype=float)
    base = frame["promise_base"].to_numpy(dtype=float)
    late_base = frame["is_late_base"].to_numpy()
    logi = frame["is_logistics"].to_numpy()

    valid = np.isfinite(price) & np.isfinite(actual) & np.isfinite(eta_new) & np.isfinite(base)
    if valid.sum() == 0:
        return None

    p, a, e, b = price[valid], actual[valid], eta_new[valid], base[valid]
    lb, lg = late_base[valid], logi[valid]

    eff = effective_days(e - b, b, t_ref, cap=cap, gain_ratio=gain_ratio)
    gmv_delta = conversion_delta(p, eff, daily_rate)

    late_new = is_late(a, e)              # 整天口径，与 GMV 表同源
    reduced = lb & ~late_new          # 原本违约、新承诺下守约
    newly = (~lb) & late_new          # 原本守约、新承诺下反被承诺坑（承诺变短导致）
    reduced_logi = int((reduced & lg).sum())
    newly_logi = int((newly & lg).sum())

    return {
        "有效订单数": int(valid.sum()),
        "平均承诺天数": float(np.mean(e)),
        "承诺变化均值": float(np.mean(e - b)),
        "转化口径ΔGMV_元": gmv_delta,                      # 正=收益
        "占同期GMV": gmv_delta / float(p.sum()),
        "改造订单数": int((np.abs(e - b) > 1e-9).sum()),    # 承诺真的变了的订单
        "延迟单量": int(late_new.sum()),
        "延迟率": float(late_new.mean()),
        "减少延迟单": int(reduced.sum()),
        "新增延迟单": int(newly.sum()),
        "净减少延迟单": int(reduced.sum() - newly.sum()),
        "物流归因_减少延迟单": reduced_logi,
        "物流归因_新增延迟单": newly_logi,
        "物流归因_挽救率": reduced_logi / logistics_total if logistics_total else np.nan,
        "保护GMV_元": float(p[reduced].sum()),             # 口碑口径，不并入净利润
        "避免1星差评": float((reduced.sum() - newly.sum()) * star1_delta),
    }


def monte_carlo(frame, eta_new, *, t_ref, cap=C.MAX_EFF_DAYS,
                gain_ratio=None, n_sim=None, seed=C.RANDOM_SEED):
    """蒙特卡洛：同时抖「单量」与「单日流失率」两个数源，给出 ΔGMV 的置信区间。

    与 precise_gmv_simulated.py 保持同一套随机机制（订单量 ±10% 有放回抽样 +
    单日流失率正态扰动截断非负），保证三方案数字可直接对照。
    """
    cap = C.MAX_EFF_DAYS if cap is None else cap
    gain_ratio = C.SHORTEN_GAIN_RATIO if gain_ratio is None else gain_ratio
    n_sim = C.N_SIMULATIONS if n_sim is None else n_sim
    price = frame["price"].to_numpy(dtype=float)
    actual = frame["actual"].to_numpy(dtype=float)
    base = frame["promise_base"].to_numpy(dtype=float)
    valid = np.isfinite(price) & np.isfinite(actual) & np.isfinite(eta_new) & np.isfinite(base)
    if valid.sum() == 0:
        return None

    p = price[valid]
    eff = effective_days(eta_new[valid] - base[valid], base[valid], t_ref,
                         cap=cap, gain_ratio=gain_ratio)
    n = len(p)
    rng = np.random.default_rng(seed)
    out = np.empty(n_sim, dtype=float)
    for i in range(n_sim):
        scale = rng.uniform(1.0 - C.VOLUME_SWING, 1.0 + C.VOLUME_SWING)
        idx = rng.integers(0, n, size=int(round(n * scale)))
        rate = max(0.0, rng.normal(C.DAILY_LOSS_RATE, C.DAILY_LOSS_STD))
        out[i] = (p[idx] * ((1.0 - rate) ** eff[idx] - 1.0)).sum()
    return out


# =========================================================
# 3. 打印工具
# =========================================================
def print_ledger(ledger, title, t_ref, logistics_total, star1_delta):
    print("\n" + "=" * 96)
    print(f"【{title}】")
    print(f"  基准：原算法 | 时效锚点 T_REF={t_ref:.2f} 天 | 物流归因超时单（挽救率分母）={logistics_total} 单")
    print(f"  1 星率差 STAR1_DELTA={star1_delta:.2%}（延迟单与非延迟单的 1 星概率之差）")
    print("  转化口径ΔGMV 正=GMV 增加，负=GMV 流失；保护GMV 为口碑价值，不与转化损益相减。")
    print("-" * 96)
    cols = ["场景", "平均承诺", "承诺变化", "ΔGMV_元", "占GMV", "延迟率", "净减少延迟单",
            "物流挽救率", "避免1星"]
    print(f"{cols[0]:<26}{cols[1]:>9}{cols[2]:>10}{cols[3]:>14}{cols[4]:>8}"
          f"{cols[5]:>9}{cols[6]:>12}{cols[7]:>11}{cols[8]:>9}")
    for _, r in ledger.iterrows():
        print(f"{pad_cn(r['场景'], 26)}{r['平均承诺天数']:>9.2f}{r['承诺变化均值']:>+10.2f}"
              f"{r['转化口径ΔGMV_元']:>14,.0f}{r['占同期GMV'] * 100:>7.2f}%{r['延迟率'] * 100:>8.2f}%"
              f"{r['净减少延迟单']:>12,d}{r['物流归因_挽救率'] * 100:>10.2f}%{r['避免1星差评']:>9.0f}")
    print("=" * 96)


# =========================================================
# 4. 可视化
# =========================================================
def pad_cn(text, width, align="left"):
    """按中文占两格的显示宽度补空格，避免控制台表格对不齐。"""
    width = max(int(width), 1)
    disp = sum(2 if ord(ch) > 0x2E7F else 1 for ch in str(text))
    pad = " " * max(width - disp, 0)
    return f"{text}{pad}" if align == "left" else f"{pad}{text}"


def breakeven_gain_ratio(frame, eta_new, *, t_ref, logistics_total, star1_delta,
                         cap=None, tol=0.005):
    """盈亏平衡点：缩短承诺带来的转化收益，至少要有「加时流失」的几成，方案才不亏钱？

    这是本次分析最关键的一个未知量——1.8%/天 只校准了加时方向，反方向完全是假设。
    与其让结论停在一个凭空设定的 gain_ratio 上，不如直接把它解出来交给业务判断：
    只要业务认为「减一天承诺」带来的转化提升不低于「加一天」流失的 X%，方案就成立。
    ΔGMV 对 gain_ratio 单调递减，用二分求解。
    """
    def value(r):
        return evaluate_scenario(frame, eta_new, t_ref=t_ref,
                                 logistics_total=logistics_total, star1_delta=star1_delta,
                                 cap=cap, gain_ratio=r)["转化口径ΔGMV_元"]

    lo, hi = 0.0, 1.0
    if value(hi) < 0:                       # 即使完全对称也亏，平衡点不存在
        return None
    if value(lo) > 0:                       # 即使收益全不算也赚，无需平衡
        return 0.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        # ΔGMV 随 gain_ratio 单调递增：mid 处已盈利说明门槛比 mid 更低，压上界
        if value(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def scenario_color(name):
    """按场景类型配色：延时补丁=橙，替换算法=蓝，组合=紫，基准=灰。"""
    if name.startswith("①"):
        return "#9e9e9e"
    if name.startswith("⑨") or name.startswith("⑩"):
        return "#8e6bb5"
    if name.startswith("⑤") or name.startswith("⑥") or name.startswith("⑦") or name.startswith("⑧"):
        return "#3d7ebf"
    return "#e08b3c"


def plot_delta_gmv(ledger, mc_summary, path):
    """转化口径 ΔGMV 对比：带 95% 置信区间误差棒，0 轴为原算法基准。"""
    names = ledger["场景"].tolist()
    means = ledger["转化口径ΔGMV_元"].to_numpy(dtype=float)
    lo, hi = np.zeros_like(means), np.zeros_like(means)
    for i, n in enumerate(names):
        if n in mc_summary.index:
            lo[i] = means[i] - mc_summary.loc[n, "P2.5"]
            hi[i] = mc_summary.loc[n, "P97.5"] - means[i]

    fig, ax = plt.subplots(figsize=C.FIGURE_SIZE, dpi=C.DPI)
    bars = ax.bar(names, means, width=0.66, color=[scenario_color(n) for n in names],
                  edgecolor="black", alpha=0.85,
                  yerr=np.vstack([np.clip(lo, 0, None), np.clip(hi, 0, None)]),
                  capsize=4, ecolor="#555555", error_kw={"elinewidth": 1.1})
    ax.axhline(0, color="black", linewidth=1.4)
    ax.set_title("ETA 策略的转化口径 GMV 损益对比（0 = 原算法基准）")
    ax.set_ylabel("ΔGMV（元，正=收益 / 负=流失）")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=28, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.bar_label(bars, labels=[f"{v:,.0f}" for v in means], padding=6, fontsize=8)
    # y 轴上限要同时容下柱体和误差棒
    upper = float(np.max(means + np.clip(hi, 0, None)))
    lower = float(np.min(means - np.clip(lo, 0, None)))
    span = upper - lower
    ax.set_ylim(lower - abs(span) * 0.10, upper + abs(span) * 0.16)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _p: f"{x / 10000:,.0f}"))
    ax.set_ylabel("ΔGMV（万元，正=收益 / 负=流失）")
    for lbl, clr in [("延时补丁（给旧算法加缓冲）", "#e08b3c"), ("替换算法（直接换新模型）", "#3d7ebf"),
                     ("组合方案", "#8e6bb5")]:
        ax.bar([], [], color=clr, label=lbl, edgecolor="black", alpha=0.85)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=C.DPI, bbox_inches="tight")
    plt.close(fig)


def plot_delay_rate(ledger, base_rate, path):
    """延迟率对比：虚线为原算法基准，柱子越短越好。"""
    names = ledger["场景"].tolist()
    rates = ledger["延迟率"].to_numpy(dtype=float) * 100
    fig, ax = plt.subplots(figsize=C.FIGURE_SIZE, dpi=C.DPI)
    bars = ax.bar(names, rates, width=0.66, color=[scenario_color(n) for n in names],
                  edgecolor="black", alpha=0.85)
    ax.axhline(base_rate * 100, color="#c62828", linestyle="--", linewidth=1.6,
               label=f"原算法基准 {base_rate * 100:.2f}%")
    ax.set_title("各 ETA 策略下的订单延迟率")
    ax.set_ylabel("延迟率（%）")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=28, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9)
    ax.set_ylim(0, max(rates.max(), base_rate * 100) * 1.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=C.DPI, bbox_inches="tight")
    plt.close(fig)


def plot_monte_carlo_distribution(mc_map, keys, path):
    """关键场景的 ΔGMV 模拟分布。"""
    keys = [k for k in keys if k in mc_map]
    ncol = len(keys)
    fig, axes = plt.subplots(1, ncol, figsize=(6 * ncol, 4.6), dpi=C.DPI)
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        arr = mc_map[key]
        ax.hist(arr, bins=C.PLOT_BINS, color=scenario_color(key), edgecolor="black", alpha=0.75)
        ax.axvline(arr.mean(), color="#c62828", linestyle="dashed", linewidth=1.8,
                   label=f"均值 {arr.mean():,.0f}")
        ax.axvline(0, color="black", linewidth=1.0)
        ax.set_title(key)
        ax.set_xlabel("ΔGMV（元）")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("模拟频次")
    fig.tight_layout()
    fig.savefig(path, dpi=C.DPI, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(sens, path):
    """收益折扣 × 等效天数封顶 的敏感性：换算法的结论有多依赖这两个未校准参数。"""
    pivot = sens.pivot(index="收益折扣", columns="等效天数封顶", values="ΔGMV_元")
    fig, ax = plt.subplots(figsize=C.FIGURE_SIZE, dpi=C.DPI)
    n = len(pivot)
    width = 0.8 / max(len(pivot.columns), 1)
    for j, colname in enumerate(pivot.columns):
        pos = np.arange(n) + j * width - (len(pivot.columns) - 1) * width / 2
        vals = pivot[colname].to_numpy(dtype=float)
        bars = ax.bar(pos, vals, width=width * 0.92, label=f"封顶={colname}", alpha=0.88,
                      edgecolor="black")
        ax.bar_label(bars, labels=[f"{v:,.0f}" for v in vals], fontsize=7, padding=3, rotation=90)
    ax.axhline(0, color="black", linewidth=1.3)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels([f"{v:g}" for v in pivot.index])
    ax.set_xlabel("缩短承诺的收益折扣 gain_ratio（1.0=与加时对称，越小越保守）")
    ax.set_ylabel(f"{C.SELECTED_MODEL} 的 ΔGMV（元）")
    ax.set_title("敏感性：结论对未校准参数的依赖程度")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=C.DPI, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 5. 主流程
# =========================================================
def main():
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    C.ASSET_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("ETA 策略 GMV 统一账本：旧算法打补丁 vs 直接换新算法")
    print("=" * 96)

    # ---------- 5.1 数据与全局口径 ----------
    frame, _route = load_inputs()
    t_ref = float(frame["promise_base"].mean())
    logistics_total = int((frame["is_late_base"] & frame["is_logistics"]).sum())
    base_rate = float(frame["is_late_base"].mean())
    star1_rate_late = float((frame.loc[frame["is_late_base"], "score"] == 1).mean())
    star1_rate_ontime = float((frame.loc[~frame["is_late_base"], "score"] == 1).mean())
    star1_delta = star1_rate_late - star1_rate_ontime

    print(f"\n数据：{len(frame):,} 单 | 同期 GMV {frame['price'].sum():,.0f} 元")
    print(f"行对齐校验通过；原算法平均承诺 {t_ref:.2f} 天，实际平均 {frame['actual'].mean():.2f} 天")
    print(f"原算法延迟率 {base_rate * 100:.2f}% | 物流归因超时单 {logistics_total:,} 单（挽救率分母）")
    print(f"1 星率：延迟单 {star1_rate_late:.2%} vs 非延迟单 {star1_rate_ontime:.2%} → 差值"
          f" {star1_delta:.2%}")

    # ---------- 5.2 主账本：点估计 + 蒙特卡洛 ----------
    print(f"\n>>> 主口径：{C.PRIMARY_SCOPE}（{C.N_SIMULATIONS:,} 次蒙特卡洛）")
    rows, mc_map, mc_rows, detail_cols = [], {}, [], {}
    for scn in C.SCENARIOS:
        eta_new = build_scenario_eta(frame, scn, use_oof=False)
        res = evaluate_scenario(frame, eta_new, t_ref=t_ref,
                                logistics_total=logistics_total, star1_delta=star1_delta)
        if res is None:
            print(f"  跳过 {scn['name']}：无可用预测值")
            continue
        rows.append({"场景": scn["name"], **res})
        arr = monte_carlo(frame, eta_new, t_ref=t_ref)
        mc_map[scn["name"]] = arr
        mc_rows.append({
            "场景": scn["name"], "均值": float(arr.mean()),
            "P2.5": float(np.percentile(arr, 2.5)), "P97.5": float(np.percentile(arr, 97.5)),
            "标准差": float(arr.std()), "负收益概率": float((arr < 0).mean()),
        })
        if C.EXPORT_ORDER_DETAIL:
            detail_cols[scn["name"]] = effective_days(
                eta_new - frame["promise_base"].to_numpy(dtype=float),
                frame["promise_base"].to_numpy(dtype=float), t_ref)

    ledger = pd.DataFrame(rows)
    mc_summary = pd.DataFrame(mc_rows).set_index("场景")
    ledger = ledger.merge(mc_summary.add_prefix("MC_"), left_on="场景", right_index=True, how="left")
    print_ledger(ledger, f"主账本（{C.PRIMARY_SCOPE}口径）", t_ref, logistics_total, star1_delta)

    ledger.to_csv(C.OUTPUT_DIR / C.LEDGER_CSV_NAME, index=False, encoding="utf-8-sig")
    mc_summary.reset_index().to_csv(C.OUTPUT_DIR / C.MONTE_CARLO_CSV_NAME, index=False,
                                    encoding="utf-8-sig")
    if C.EXPORT_ORDER_DETAIL:
        detail = pd.DataFrame({
            "下单时间": frame["order_time"], "总价格": frame["price"],
            "原算法承诺": frame["promise_base"], "实际总周期": frame["actual"],
            "是否延迟_原算法": frame["is_late_base"], "是否物流责任": frame["is_logistics"],
        })
        for k, v in detail_cols.items():
            detail[f"{k}_等效天数变化"] = np.round(v, 3)
        detail.to_csv(C.OUTPUT_DIR / C.ORDER_DETAIL_CSV_NAME, index=False, encoding="utf-8-sig")

    # ---------- 5.3 稳健性：三口径 ----------
    print("\n>>> 稳健性：新模型的延迟率是否被样本内乐观高估了？")
    cut = frame["order_time"].quantile(1 - C.TEST_SIZE)
    test_mask = (frame["order_time"] >= cut).to_numpy()
    oof_mask = frame[[f"oof_{m}" for m in C.MODEL_SOURCE_COLS]].notna().all(axis=1).to_numpy() \
        if f"oof_{C.MODEL_SOURCE_COLS[0]}" in frame.columns else np.zeros(len(frame), bool)

    base_scn = C.SCENARIOS[0]
    rob_rows = []
    scope_defs = {
        "全量": np.ones(len(frame), bool),
        "时间外测试期": test_mask,
        "OOF折外": oof_mask,
    }
    for scope, mask in scope_defs.items():
        if mask.sum() == 0:
            continue
        sub = frame.loc[mask].copy()
        sub_base_rate = float(sub["is_late_base"].mean())
        sub_logi = int((sub["is_late_base"] & sub["is_logistics"]).sum())
        scn_list = [base_scn] + [{"name": C.MODEL_DISPLAY_NAME[m], "source": m, "extend": "none"}
                                 for m in C.ROBUSTNESS_MODELS]
        for scn in scn_list:
            use_oof = scope == "OOF折外" and scn["source"] != "原算法"
            eta_new = build_scenario_eta(sub, scn, use_oof=use_oof)
            res = evaluate_scenario(sub, eta_new, t_ref=t_ref,
                                    logistics_total=sub_logi if sub_logi else logistics_total,
                                    star1_delta=star1_delta)
            if res is None:
                continue
            rob_rows.append({
                "口径": scope, "样本数": int(mask.sum()),
                "口径内延迟率_基准": sub_base_rate,
                "场景": scn["name"],
                "平均承诺天数": res["平均承诺天数"], "延迟率": res["延迟率"],
                "转化口径ΔGMV_元": res["转化口径ΔGMV_元"], "占同期GMV": res["占同期GMV"],
                "净减少延迟单": res["净减少延迟单"],
            })
    robustness = pd.DataFrame(rob_rows)
    print("\n" + "-" * 96)
    print(f"{'口径':<14}{'样本数':>8}{'基准延迟率':>11}{'方案':<20}{'延迟率':>9}{'ΔGMV_元':>14}{'占GMV':>8}")
    for _, r in robustness.iterrows():
        print(f"{pad_cn(r['口径'], 12)}{r['样本数']:>8,d}{r['口径内延迟率_基准'] * 100:>10.2f}%"
              f"{pad_cn(r['场景'], 24)}{r['延迟率'] * 100:>8.2f}%{r['转化口径ΔGMV_元']:>14,.0f}"
              f"{r['占同期GMV'] * 100:>7.2f}%")
    print("-" * 96)
    robustness.to_csv(C.OUTPUT_DIR / C.ROBUSTNESS_CSV_NAME, index=False, encoding="utf-8-sig")

    # ---------- 5.4 与原报告三方案的对照（验证口径没走样） ----------
    print("\n>>> 与原报告《ETA 动态延时策略》的三方案对照（确认口径一致）")
    print(f"{'方案':<26}{'改造订单数':>10}{'物流归因救回':>13}{'挽救率':>9}{'GMV流失':>13}{'单救成本':>10}")
    for scn in C.SCENARIOS[1:4]:
        row = ledger.loc[ledger["场景"] == scn["name"]]
        if row.empty:
            continue
        r = row.iloc[0]
        cnt = max(int(r["物流归因_减少延迟单"]), 1)
        print(f"{pad_cn(scn['name'], 26)}{int(r['改造订单数']):>10,d}{int(r['物流归因_减少延迟单']):>13,d}"
              f"{r['物流归因_挽救率'] * 100:>8.2f}%{-r['转化口径ΔGMV_元']:>13,.0f}"
              f"{-r['转化口径ΔGMV_元'] / cnt:>10,.0f}")
    print("  原报告数值：A 全量+2天 救回 1,296 / 26.89% / GMV流失 46.45万 / 单救 358 元；"
          "B 仅物流单 3.49万 / 27 元；C 线路阶梯 救回 1,139 / 23.63% / 32.53万")
    print("  口径差异说明：本表走点估计，原报告走 10,000 次蒙特卡洛均值，两者允许有 <1% 出入；"
          "延迟判定改用「四舍五入到整天」的统一尺子（更贴合 GMV 表送达状态）。")

    # ---------- 5.5 敏感性：两个未校准参数 ----------
    print(f"\n>>> 敏感性：{C.MODEL_DISPLAY_NAME[C.SELECTED_MODEL]} 的结论有多依赖未校准参数？")
    sens_rows = []
    scn_sel = [s for s in C.SCENARIOS if s["source"] == C.SELECTED_MODEL and s["extend"] == "none"][0]
    eta_sel = build_scenario_eta(frame, scn_sel, use_oof=False)
    for ratio in C.SHORTEN_GAIN_RATIO_GRID:
        for cap in C.EFF_CAP_GRID:
            res = evaluate_scenario(frame, eta_sel, t_ref=t_ref,
                                    logistics_total=logistics_total, star1_delta=star1_delta,
                                    cap=cap, gain_ratio=ratio)
            sens_rows.append({
                "模型": C.SELECTED_MODEL, "收益折扣": ratio,
                "等效天数封顶": "不封顶" if np.isinf(cap) else f"{cap:g}天",
                "ΔGMV_元": res["转化口径ΔGMV_元"], "占同期GMV": res["占同期GMV"],
                "延迟率": res["延迟率"], "净减少延迟单": res["净减少延迟单"],
            })
    sens = pd.DataFrame(sens_rows)
    print(f"{'收益折扣':>8}{'等效天数封顶':>14}{'ΔGMV_元':>15}{'占GMV':>9}{'延迟率':>9}")
    for _, r in sens.iterrows():
        print(f"{r['收益折扣']:>8g}{r['等效天数封顶']:>14}{r['ΔGMV_元']:>15,.0f}"
              f"{r['占同期GMV'] * 100:>8.2f}%{r['延迟率'] * 100:>8.2f}%")
    # 盈亏平衡：把「未校准参数」从假设变成可判断题
    print("\n>>> 盈亏平衡：必须相信「减时提转化」到什么程度，方案才成立？")
    for scn in C.SCENARIOS:
        if scn["source"] == "原算法":
            continue
        eta_new = build_scenario_eta(frame, scn, use_oof=False)
        be14 = breakeven_gain_ratio(frame, eta_new, t_ref=t_ref,
                                    logistics_total=logistics_total, star1_delta=star1_delta,
                                    cap=C.MAX_EFF_DAYS)
        beunc = breakeven_gain_ratio(frame, eta_new, t_ref=t_ref,
                                     logistics_total=logistics_total, star1_delta=star1_delta,
                                     cap=np.inf)
        def fmt(v):
            if v is None:
                return "对称也亏"
            if v <= 0.01:
                return "无条件成立"
            if v >= 0.99:
                return "需近乎完全对称"
            return f"{v:.2f}"

        print(f"  {pad_cn(scn['name'], 34)}封顶{C.MAX_EFF_DAYS:g}天：{fmt(be14):>14}"
              f"    不封顶：{fmt(beunc):>14}")
    print("  读法：填 X.XX = 只要「缩短承诺的转化增益」不低于「延长承诺的转化流失」的 X%，该方案净赚；"
          "填「无条件成立」表示连收益都不算也赚钱。")

    sens.to_csv(C.OUTPUT_DIR / C.SENSITIVITY_CSV_NAME, index=False, encoding="utf-8-sig")

    # ---------- 5.5 可视化 ----------
    p1 = C.ASSET_DIR / "gmv_delta_comparison.png"
    p2 = C.ASSET_DIR / "gmv_delay_rate_comparison.png"
    p3 = C.ASSET_DIR / "gmv_monte_carlo_distribution.png"
    p4 = C.ASSET_DIR / "gmv_sensitivity.png"
    plot_delta_gmv(ledger, mc_summary, p1)
    plot_delay_rate(ledger, base_rate, p2)
    plot_monte_carlo_distribution(mc_map, [C.SCENARIOS[1]["name"], C.SCENARIOS[6]["name"],
                                           C.SCENARIOS[8]["name"]], p3)
    plot_sensitivity(sens, p4)

    # ---------- 5.6 结论摘要 ----------
    best_delay = ledger.loc[ledger["场景"] != C.SCENARIOS[0]["name"], "延迟率"].idxmin()
    best_gmv = ledger["转化口径ΔGMV_元"].idxmax()
    sel_row = ledger.loc[ledger["场景"].str.contains(C.SELECTED_MODEL, regex=False)].head(1)
    print("\n" + "=" * 96)
    print("【结论摘要】")
    print(f"  · 转化口径收益最高：{ledger.loc[best_gmv, '场景']} "
          f"{ledger.loc[best_gmv, '转化口径ΔGMV_元']:,.0f} 元"
          f"（占同期 GMV {ledger.loc[best_gmv, '占同期GMV'] * 100:.2f}%）")
    print(f"  · 延迟率最低：{ledger.loc[best_delay, '场景']} "
          f"{ledger.loc[best_delay, '延迟率'] * 100:.2f}%（基准 {base_rate * 100:.2f}%）")
    if len(sel_row):
        r = sel_row.iloc[0]
        print(f"  · 选定模型 {C.SELECTED_MODEL}（部署口径）：ΔGMV {r['转化口径ΔGMV_元']:,.0f} 元，"
              f"延迟率 {r['延迟率'] * 100:.2f}%，净减少延迟单 {r['净减少延迟单']:,}，"
              f"避免 1 星差评约 {r['避免1星差评']:.0f} 个")
    # 三口径对照：把「样本内乐观」这条底线讲清楚
    print("\n  ·【诚实口径】同一模型在三个样本上的延迟率（避免被部署口径的乐观数字误导）：")
    sel_label = C.MODEL_DISPLAY_NAME[C.SELECTED_MODEL]
    for scope in C.SCOPES:
        sub = robustness[robustness["口径"] == scope]
        if sub.empty:
            continue
        b = float(sub["口径内延迟率_基准"].iloc[0])
        m = sub[sub["场景"] == sel_label]
        line = f"      {scope:<8}基准 {b * 100:5.2f}%"
        if len(m):
            line += (f" → {sel_label} {float(m['延迟率'].iloc[0]) * 100:5.2f}%"
                     f"（ΔGMV {float(m['转化口径ΔGMV_元'].iloc[0]):>12,.0f} 元）")
        print(line)
    print("      全量口径里 model 见过这些订单，延迟率是被低估的；OOF 折外才是持续部署后的真实水平。")
    print("=" * 96)
    print(f"\n表格输出目录：{C.OUTPUT_DIR}")
    print(f"图表输出目录：{C.ASSET_DIR}")


if __name__ == "__main__":
    main()
