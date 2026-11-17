# %% 1. 导入库
"""
Transformer + VCN 对比模型：训练脚本（对标 lgb_model.py）

设计思路
--------
LightGBM 直接吃「已经聚合好的静态特征表」，而本模型显式建模时间维度：
  * 对每条订单，以它下单之前若干个「时间桶」的历史快照构成一个序列；
  * Transformer 编码器读取整段序列，捕捉平台物流状态的长期演化（长时序依赖）；
  * 多尺度 1D 卷积（VCN）在序列上滑动，捕捉最近几个时间片的局部突变（局部感受野）；
  * 两路表征拼接订单当期的静态特征，输出对数配送周期。

损失口径与 lgb_model.py 完全一致：L1 基线 + 非对称 Pseudo-Huber（低估惩罚加倍），
Pseudo-Huber 模型同样在「中心化后的对数目标」上训练，预测时把中心值加回。

本文件也是公共模块：Transformer_CNN_model_val.py / _ooc.py 直接 import 这里的
序列构造、模型定义、损失与评估函数，避免两处重复实现。训练主流程放在 __main__ 保护内，
import 时不会触发训练。
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:  # 轻量环境（如 pt_study）未装 seaborn 时，自动降级到 matplotlib 画核密度
    sns = None
    _HAS_SEABORN = False
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_prepare import df as feature_data  # 与 lgb_model.py 同源的特征表（英文列名）
from config import CONFIG

# %% 2. 全局配置
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

RANDOM_STATE = CONFIG["random_state"]
FULL_START_TIME = pd.Timestamp(CONFIG["full_start_date"])
SEVERE_UNDER_DAYS = CONFIG["severe_under_days"]
BASE_DIR = CONFIG["base_dir"]
OUTPUT_DIR = Path(CONFIG["output_dir"])
ASSET_DIR = Path(CONFIG["asset_dir"])
MODEL_DIR = Path(CONFIG["model_dir"])
for d in (OUTPUT_DIR, ASSET_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

def resolve_device():
    """
    自动设备判断：有可用 CUDA 则用 GPU，否则回退 CPU。
    - config['device'] 支持 "auto" / "cuda" / "cuda:N" / "cpu"；
    - 指定的 GPU 序号不存在（或驱动不可用）时安全回退，不会因配置写死而报错；
    - 同时固定 CPU / CUDA 随机种子，保证不同设备上的实验可复现。
    """
    preferred = str(CONFIG.get("device", "auto")).strip().lower()
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # "auto" 或 "cuda[:N]" 都表示：有 GPU 就优先用 GPU
    want_cuda = preferred == "auto" or preferred.startswith("cuda")
    if want_cuda and torch.cuda.is_available():
        count = torch.cuda.device_count()
        idx = 0
        if ":" in preferred:
            try:
                idx = int(preferred.split(":", 1)[1])
            except ValueError:
                idx = 0
        if idx < 0 or idx >= count:
            print(f"[设备] 配置的 {preferred} 不存在（可见 {count} 张卡），自动改用 cuda:0")
            idx = 0
        device = torch.device(f"cuda:{idx}")
        torch.cuda.manual_seed_all(RANDOM_STATE)
        torch.backends.cudnn.benchmark = True
        name = torch.cuda.get_device_name(idx)
        mem = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
        print(f"[设备] 使用 GPU：{device} · {name} · {mem:.1f} GB（可见 {count} 张卡）")
        return device

    if want_cuda:
        print(f"[设备] 配置请求 {preferred}，但未检测到可用 CUDA，自动回退 CPU")
    else:
        print("[设备] 按配置使用 CPU")
    return torch.device("cpu")


DEVICE = resolve_device()
PREFIX = CONFIG["tc_output_prefix"]          # 输出文件前缀，如 tcnn_
MODEL_PREFIX = "TransCNN"                    # 模型名前缀，如 TransCNN_L1

# 类别特征与排除列：与 classifier.py / lgb_model.py 保持同一口径
CAT_FEATURES = ["product_type", "payment_method", "seller_state", "customer_state"]
EXCLUDE_COLS = ["order_time", "seller_id", "seller_city", "customer_city", "_row_id",
                "total_delivery_time", "log_delivery_time", "estimated_delivery_time"]
SNAP_COLS = ["cnt", "avg_delivery", "p90_delivery", "avg_estimated", "late_rate", "avg_gap"]
SNAP_DIM = len(SNAP_COLS)


# %% 3. 序列构造：历史时间桶快照
def build_sequence_features(data, seq_steps=None, bucket_days=None, seq_mode=None,
                            t0_override=None, verbose=True):
    """
    把已按 order_time 排序的特征表转成「历史序列 + 当期静态特征」中的历史序列部分。

    参数
    ----
    data : DataFrame，需含 order_time / total_delivery_time / estimated_delivery_time
           （route 模式还需 seller_state / customer_state）
    seq_steps  : 历史时间片个数（config.tc_seq_steps）
    bucket_days: 每个时间片跨多少天（config.tc_bucket_days）
    seq_mode   : 'global' 全平台快照（稠密，默认）；'route' 按线路聚合（更贴近业务但更稀疏）
    t0_override: 时间桶原点。验证/折外必须传入训练时保存的 meta['seq_t0']，
                 否则桶编号整体偏移，序列分布与训练不一致。

    返回
    ----
    seq        : float32 array [N, seq_steps, SNAP_DIM]
    bucket_idx : int array [N]，每条订单所在的时间桶编号
    snap_mean  : 每个快照维度的全局均值，用于填补「历史里没有订单的空桶」
    t0         : 本次使用的时间桶原点（Timestamp）
    """
    seq_steps = seq_steps or CONFIG["tc_seq_steps"]
    bucket_days = bucket_days or CONFIG["tc_bucket_days"]
    seq_mode = seq_mode or CONFIG["tc_seq_mode"]

    d = data.reset_index(drop=True).copy()
    t0 = pd.Timestamp(t0_override) if t0_override is not None else d["order_time"].min()
    days_since = (d["order_time"] - t0).dt.total_seconds() / 86400.0
    bucket_idx = np.floor(days_since / bucket_days).astype(int).to_numpy()
    d["_bucket"] = bucket_idx
    # 是否延迟：用于快照里的 late_rate（实际 > 原算法承诺）
    d["_late"] = (d["total_delivery_time"] > d["estimated_delivery_time"]).astype(float)
    d["_gap"] = d["total_delivery_time"] - d["estimated_delivery_time"]

    def _agg(df_part, keys):
        g = df_part.groupby(keys, sort=True)
        out = g.agg(
            cnt=("total_delivery_time", "size"),
            avg_delivery=("total_delivery_time", "mean"),
            p90_delivery=("total_delivery_time", lambda s: s.quantile(0.9)),
            avg_estimated=("estimated_delivery_time", "mean"),
            late_rate=("_late", "mean"),
            avg_gap=("_gap", "mean"),
        )
        return out

    if seq_mode == "global":
        snap = _agg(d, ["_bucket"])
        # 空桶用全平台均值填补
        snap_mean = snap[SNAP_COLS].mean(axis=0).to_numpy(dtype=np.float32)
        lookup = {int(b): snap.loc[b, SNAP_COLS].to_numpy(dtype=np.float32) for b in snap.index}
        get_row = lambda b: lookup.get(int(b), snap_mean)
    elif seq_mode == "route":
        snap = _agg(d, ["_bucket", "seller_state", "customer_state"])
        snap_mean = snap[SNAP_COLS].mean(axis=0).to_numpy(dtype=np.float32)
        lookup = {(int(k[0]), str(k[1]), str(k[2])):
                  snap.loc[k, SNAP_COLS].to_numpy(dtype=np.float32) for k in snap.index}
        ss = d["seller_state"].astype(str).to_numpy()
        cs = d["customer_state"].astype(str).to_numpy()

        def get_row_for(i, b):
            return lookup.get((int(b), ss[i], cs[i]), snap_mean)
        get_row = get_row_for
    else:
        raise ValueError(f"未知的 tc_seq_mode：{seq_mode}，可选 'global' / 'route'")

    N = len(d)
    seq = np.empty((N, seq_steps, SNAP_DIM), dtype=np.float32)
    for i in range(N):
        b = bucket_idx[i]
        # 历史 = 已结束的桶 [b-seq_steps, ..., b-1]，正序排列；绝不使用当前桶，避免结果泄露
        for j, hist_b in enumerate(range(b - seq_steps, b)):
            if seq_mode == "global":
                seq[i, j] = get_row(hist_b)
            else:
                seq[i, j] = get_row(i, hist_b)

    if verbose:
        print(f"序列构造完成（{seq_mode}）：{N} 条 × {seq_steps} 个时间片 × {SNAP_DIM} 维，"
              f"每片 {bucket_days} 天，历史窗口约 {seq_steps * bucket_days} 天，原点 {t0}")
    return seq, bucket_idx, snap_mean, t0


def build_static_features(data, cat_categories=None):
    """
    构造当期静态特征：数值特征 + 类别 one-hot。
    类别集合可在训练集上 fit 一次后传入，保证训练/验证/折外维度一致。

    返回 (static: float64 array, cat_categories: dict)
    """
    numeric_cols = [c for c in data.columns if c not in EXCLUDE_COLS + CAT_FEATURES]
    numeric_cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(data[c])]
    part = [data[numeric_cols].astype(float)]

    cats = {}
    for c in CAT_FEATURES:
        vals = data[c].astype("object").fillna("未知")
        categories = cat_categories[c] if cat_categories else sorted(vals.unique().tolist())
        cats[c] = categories
        dummy = pd.get_dummies(pd.Categorical(vals, categories=categories), prefix=c, dtype=float)
        part.append(dummy)
    static = pd.concat(part, axis=1).to_numpy(dtype=np.float64)
    return static, cats


# %% 4. 模型定义
class TransformerVCN(nn.Module):
    """Transformer（长时序依赖）+ 多尺度 1D 卷积 VCN（局部感受野）+ 静态特征融合。"""

    def __init__(self, seq_dim=SNAP_DIM, static_dim=10, d_model=None, nhead=None,
                 num_layers=None, dim_feedforward=None, dropout=None,
                 vcn_channels=None, kernel_sizes=None, static_hidden=None,
                 fusion_hidden=None, max_seq_len=None):
        super().__init__()
        d_model = d_model or CONFIG["tc_d_model"]
        nhead = nhead or CONFIG["tc_nhead"]
        num_layers = num_layers or CONFIG["tc_num_layers"]
        dim_feedforward = dim_feedforward or CONFIG["tc_dim_feedforward"]
        dropout = CONFIG["tc_dropout"] if dropout is None else dropout
        vcn_channels = vcn_channels or CONFIG["tc_vcn_channels"]
        kernel_sizes = kernel_sizes or CONFIG["tc_vcn_kernel_sizes"]
        static_hidden = static_hidden or CONFIG["tc_static_hidden"]
        fusion_hidden = fusion_hidden or CONFIG["tc_fusion_hidden"]
        max_seq_len = max_seq_len or CONFIG["tc_seq_steps"]

        if d_model % nhead != 0:
            raise ValueError(f"tc_d_model={d_model} 必须能被 tc_nhead={nhead} 整除")

        # 序列分支：投影 + 可学习位置编码 + Transformer 编码器
        self.seq_proj = nn.Linear(seq_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.01)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # VCN 分支：多尺度卷积，核越大感受野越宽；全局最大池化抽取每种尺度的局部模式
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, vcn_channels, k, padding=k // 2) for k in kernel_sizes
        ])

        # 静态分支
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, static_hidden), nn.GELU(),
            nn.Dropout(dropout),
        )
        vcn_out = vcn_channels * len(kernel_sizes)
        self.fusion = nn.Sequential(
            nn.Linear(vcn_out + static_hidden, fusion_hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, seq, static):
        B, L, _ = seq.shape
        x = self.seq_proj(seq) + self.pos_emb[:, :L, :]
        x = self.transformer(x)              # [B, L, d_model]
        x = x.transpose(1, 2)                # [B, d_model, L] 供 1D 卷积使用
        parts = [F.adaptive_max_pool1d(conv(x), 1).squeeze(-1) for conv in self.convs]
        seq_repr = torch.cat(parts, dim=-1)  # [B, vcn_channels*K]
        s = self.static_proj(static)         # [B, static_hidden]
        return self.fusion(torch.cat([seq_repr, s], dim=-1)).squeeze(-1)  # [B] 对数周期


def pseudo_huber_loss(pred, target, under_weight, over_weight=1.0, delta=0.2, cap=20.0):
    """
    非对称 Pseudo-Huber 损失（与 lgb_model.py 的梯度同构）。
    pred-target < 0 表示低估，乘以更大的 under_weight；Pseudo-Huber 使残差很大时梯度饱和，
    再对权重做 cap 截断，防止极端倍率把梯度放大到不稳定。
    """
    err = pred - target
    weight = torch.where(err < 0, torch.as_tensor(under_weight, device=err.device),
                         torch.as_tensor(over_weight, device=err.device))
    weight = torch.clamp(weight, max=cap)
    denom = torch.sqrt(1.0 + (err / delta) ** 2)
    return (weight * delta ** 2 * (denom - 1.0)).mean()


# %% 5. 公共评估与绘图函数（val / ooc 直接复用，口径与 lgb_model.py 完全一致）
PCT_COLS = ["early_delivery_rate", "delay_rate", "exact_rate", "severe_overtime_rate"]


def display_name(model_name):
    return "原算法" if model_name == "Original_Algorithm" else model_name


def kdeplot(values, label, linewidth=2.0):
    """核密度曲线：有 seaborn 用 seaborn，否则用 scipy/matplotlib 降级实现。"""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return
    if _HAS_SEABORN:
        sns.kdeplot(values, label=label, linewidth=linewidth)
        return
    # 降级：用高斯核密度估计后画折线（scipy 缺失时退回直方图概率密度）
    try:
        from scipy.stats import gaussian_kde
        grid = np.linspace(values.min(), values.max(), 200)
        density = gaussian_kde(values)(grid)
        plt.plot(grid, density, label=label, linewidth=linewidth)
    except ImportError:
        plt.hist(values, bins=80, density=True, histtype="step",
                 label=label, linewidth=linewidth)



def evaluate_predictions(predictions, scope_name):
    """预测精度、提前送达率与延迟风险；比例保存为 0~1。"""
    rows = []
    for model_name, group in predictions.groupby("model", sort=False):
        actual = group["actual_days"].to_numpy(dtype=float)
        pred = group["pred_days"].to_numpy(dtype=float)
        valid = np.isfinite(actual) & np.isfinite(pred) & (actual > 0) & (pred > 0)
        actual, pred = actual[valid], pred[valid]
        if len(actual) == 0:
            continue
        error = pred - actual
        rows.append({
            "model": model_name, "sample_count": len(actual),
            "MAE_days": mean_absolute_error(actual, pred),
            "RMSE_days": np.sqrt(mean_squared_error(actual, pred)),
            "pred_mean_days": np.mean(pred), "actual_mean_days": np.mean(actual),
            "mean_error_days": np.mean(error),
            "early_delivery_rate": np.mean(error > 0), "delay_rate": np.mean(error < 0),
            "exact_rate": np.mean(np.isclose(pred, actual, rtol=1e-9, atol=1e-9)),
            "severe_overtime_rate": np.mean(error < -SEVERE_UNDER_DAYS),
        })
    if not rows:
        raise ValueError(f"{scope_name}没有有效预测结果。")
    metrics_df = pd.DataFrame(rows).sort_values("MAE_days").reset_index(drop=True)
    show = metrics_df.copy()
    show[PCT_COLS] *= 100
    print(f"\n========== {scope_name}：预期总周期评估 ==========")
    print(f"严重超时率：真实比预期多超过 {SEVERE_UNDER_DAYS:.1f} 天的订单占比；比例单位 %。")
    print(show.round(3).to_string(index=False))
    return metrics_df


def save_distribution_table(predictions, scope_name, file_name):
    rows = []

    def add(name, values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        if len(values) == 0:
            return
        rows.append({
            "数据来源": name, "样本数": len(values), "均值": np.mean(values),
            "标准差": np.std(values), "最小值": np.min(values),
            "P25": np.quantile(values, .25), "中位数": np.median(values),
            "P75": np.quantile(values, .75), "P90": np.quantile(values, .90),
            "P95": np.quantile(values, .95), "最大值": np.max(values),
        })

    first_group = next(iter(predictions.groupby("model", sort=False)))[1]
    add(f"{scope_name}真实总运送周期", first_group["actual_days"].to_numpy(dtype=float))
    for name, group in predictions.groupby("model", sort=False):
        add(f"{display_name(name)}预期总周期", group["pred_days"].to_numpy(dtype=float))
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / file_name, index=False, encoding="utf-8-sig")
    return df


def plot_distribution(predictions, scope_name, figure_name):
    first_group = next(iter(predictions.groupby("model", sort=False)))[1]
    actual = first_group["actual_days"].to_numpy(dtype=float)
    actual = actual[np.isfinite(actual) & (actual > 0)]
    plt.figure(figsize=CONFIG["figure_size"])
    kdeplot(actual, "真实总运送周期", linewidth=2.8)
    for name, group in predictions.groupby("model", sort=False):
        v = group["pred_days"].to_numpy(dtype=float)
        v = v[np.isfinite(v) & (v > 0)]
        if len(v) > 1:
            kdeplot(v, display_name(name), linewidth=2)
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
    plt.figure(figsize=CONFIG["figure_size"])
    for name, group in predictions.groupby("model", sort=False):
        a = group["actual_days"].to_numpy(dtype=float)
        p = group["pred_days"].to_numpy(dtype=float)
        ok = np.isfinite(a) & np.isfinite(p) & (a > 0) & (p > 0)
        err = p[ok] - a[ok]
        if len(err) > 1:
            kdeplot(err, display_name(name), linewidth=2)
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
    names = [display_name(m) for m in metrics_df["model"].tolist()]
    rates = metrics_df["delay_rate"].to_numpy() * 100
    fig, ax = plt.subplots(figsize=(13, 6), dpi=CONFIG["figure_dpi"])
    bars = ax.bar(names, rates, width=0.65)
    ax.set_title(f"{scope_name}：不同算法的配送延迟率对比")
    ax.set_xlabel("模型名称")
    ax.set_ylabel("延迟率（%）")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9)
    ax.set_ylim(0, max(rates) * 1.15 if len(rates) else 1)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / figure_name, dpi=CONFIG["figure_dpi"], bbox_inches="tight")
    plt.show()
    plt.close(fig)


def save_summary(metrics_df, scope_name, file_name):
    cols = ["model", "sample_count", "MAE_days", "RMSE_days", "pred_mean_days",
            "actual_mean_days", "mean_error_days"] + PCT_COLS
    out = metrics_df[cols].copy()
    out[PCT_COLS] *= 100
    out.to_csv(OUTPUT_DIR / file_name, index=False, encoding="utf-8-sig")
    return out


# %% 6. 训练单模型
def train_one_model(model_name, loss_kind, under_weight, center,
                    seq_tr, static_tr, y_tr, seq_te, static_te,
                    static_dim, epochs=None, log_interval=None):
    """训练一个 TransCNN 模型，返回 (模型, 测试集对数预测[已加回中心])。"""
    epochs = epochs or CONFIG["tc_epochs"]
    log_interval = log_interval or CONFIG["tc_log_interval"]

    t_seq_tr = torch.tensor(seq_tr, dtype=torch.float32)
    t_static_tr = torch.tensor(static_tr, dtype=torch.float32)
    t_y_tr = torch.tensor(y_tr, dtype=torch.float32)          # 已中心化（Pseudo-Huber）或原始（L1）
    t_seq_te = torch.tensor(seq_te, dtype=torch.float32).to(DEVICE)
    t_static_te = torch.tensor(static_te, dtype=torch.float32).to(DEVICE)

    loader = DataLoader(TensorDataset(t_seq_tr, t_static_tr, t_y_tr),
                        batch_size=CONFIG["tc_batch_size"], shuffle=True, drop_last=False,
                        pin_memory=(DEVICE.type == "cuda"))

    model = TransformerVCN(static_dim=static_dim).to(DEVICE)
    if loss_kind == "l1":
        criterion = nn.L1Loss()
    else:
        criterion = lambda p, t: pseudo_huber_loss(
            p, t, under_weight=under_weight, over_weight=CONFIG["over_weight"],
            delta=CONFIG["tc_delta"], cap=CONFIG["tc_weight_cap"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["tc_lr"],
                                  weight_decay=CONFIG["tc_weight_decay"])
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=CONFIG["tc_warmup_epochs"])
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - CONFIG["tc_warmup_epochs"], 1), eta_min=CONFIG["tc_min_lr"])
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[CONFIG["tc_warmup_epochs"]])

    print(f"\n>>> 训练 {model_name} | 损失={loss_kind} | 参数量="
          f"{sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for sb, stb, yb in loader:
            sb, stb, yb = (sb.to(DEVICE, non_blocking=True),
                           stb.to(DEVICE, non_blocking=True),
                           yb.to(DEVICE, non_blocking=True))
            optimizer.zero_grad()
            loss = criterion(model(sb, stb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["tc_grad_clip"])
            optimizer.step()
            total_loss += loss.item() * len(yb)
        scheduler.step()
        if epoch == 1 or epoch % log_interval == 0 or epoch == epochs:
            print(f"  epoch {epoch:3d}/{epochs} | lr {scheduler.get_last_lr()[0]:.2e} "
                  f"| loss {total_loss / len(t_y_tr):.4f}")

    model.eval()
    with torch.no_grad():
        pred_log = model(t_seq_te, t_static_te).cpu().numpy()
    return model, pred_log + center   # 加回中心，得到与 LGB 同口径的对数预测


def normalize_fit_transform(train_arr, *other_arrs):
    """用训练集统计量做中位数填补 + 标准化，返回 (变换后的各数组, meta)。"""
    fill = np.nanmedian(train_arr, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)

    def _fill(a):
        return np.where(np.isfinite(a), a, fill)

    tr = _fill(train_arr)
    mean = tr.mean(axis=0)
    std = tr.std(axis=0)
    std[std < 1e-8] = 1.0
    out = [((tr - mean) / std).astype(np.float32)]
    for a in other_arrs:
        out.append(((_fill(a) - mean) / std).astype(np.float32))
    return out, {"fill": fill, "mean": mean, "std": std}


def normalize_apply(arr, stats):
    a = np.where(np.isfinite(arr), arr, stats["fill"])
    return ((a - stats["mean"]) / stats["std"]).astype(np.float32)


def days_from_log(pred_log):
    """对数周期 -> 天，clip 与 lgb_model.py 一致。"""
    return np.exp(np.clip(pred_log, -20, 20))


# %% 7. 训练主流程
if __name__ == "__main__":
    print(f"计算设备：{DEVICE} | 输出前缀：{PREFIX}")

    # ---- 数据：与 lgb_model.py 相同的清洗与口径 ----
    data = feature_data.copy()
    data["order_time"] = pd.to_datetime(data["order_time"], errors="coerce")
    required = ["order_time", "total_delivery_time", "log_delivery_time", "estimated_delivery_time"]
    for c in required:
        if c not in data.columns:
            raise ValueError(f"特征表缺少必要字段：{c}")
    for c in required[1:]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(subset=required).copy()
    data = data[(data["total_delivery_time"] > 0) & (data["estimated_delivery_time"] > 0)
                & np.isfinite(data["log_delivery_time"])].copy()
    data = data[data["order_time"] >= FULL_START_TIME].copy()
    data = data.sort_values("order_time").reset_index(drop=True)
    print(f"建模数据：{len(data)} 行 | 时间范围 {data['order_time'].min()} ~ {data['order_time'].max()}")

    # ---- 构造序列与静态特征 ----
    seq, bucket_idx, snap_mean, seq_t0 = build_sequence_features(data)
    static, cat_categories = build_static_features(data)

    # ---- 时间切分（与 lgb_model.py 完全相同的切分点）----
    cutoff_time = data["order_time"].quantile(1 - CONFIG["test_size"])
    train_mask = (data["order_time"] < cutoff_time).to_numpy()
    test_mask = ~train_mask
    print(f"切分点 {cutoff_time} | 训练 {train_mask.sum()} 条 | 测试 {test_mask.sum()} 条")

    y_log = data["log_delivery_time"].to_numpy(dtype=float)
    y_test_days = data.loc[test_mask, "total_delivery_time"].to_numpy(dtype=float)
    original_estimated_days = data.loc[test_mask, "estimated_delivery_time"].to_numpy(dtype=float)

    # 标准化：只用训练集 fit，避免测试集信息泄漏。序列与静态特征各自一套统计量。
    seq_flat = seq.reshape(-1, SNAP_DIM)
    (seq_tr_flat, seq_te_flat), seq_stats = normalize_fit_transform(
        seq[train_mask].reshape(-1, SNAP_DIM), seq[test_mask].reshape(-1, SNAP_DIM))
    seq_tr = seq_tr_flat.reshape(train_mask.sum(), CONFIG["tc_seq_steps"], SNAP_DIM)
    seq_te = seq_te_flat.reshape(test_mask.sum(), CONFIG["tc_seq_steps"], SNAP_DIM)

    (static_tr, static_te, static_all), static_stats = normalize_fit_transform(
        static[train_mask], static[test_mask], static)

    # ---- 模型配置：L1 + 各倍率 Pseudo-Huber（与 lgb_model.py 的命名对齐）----
    target_center = float(np.median(y_log[train_mask]))
    configs = [("L1", "l1", 0.0, 0.0)]
    for w in CONFIG["tc_under_weights"]:
        configs.append((f"PseudoHuber_Under{w:g}x", "pseudo_huber", target_center, w))

    prediction_frames = [pd.DataFrame({
        "model": "Original_Algorithm", "actual_days": y_test_days,
        "pred_days": original_estimated_days, "original_estimated_days": original_estimated_days,
    })]
    model_records = []

    for suffix, loss_kind, center, under_w in configs:
        name = f"{MODEL_PREFIX}_{suffix}"
        y_tr = y_log[train_mask] - center
        model, pred_log = train_one_model(
            name, loss_kind, under_w, center,
            seq_tr, static_tr, y_tr, seq_te, static_te,
            static_dim=static.shape[1])
        pred_days = days_from_log(pred_log)

        # 模型权重直接存中文路径（torch.save 无 LightGBM 的 C API 编码问题）
        model_path = MODEL_DIR / f"{name}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存：{model_path}")

        prediction_frames.append(pd.DataFrame({
            "model": name, "actual_days": y_test_days,
            "pred_days": pred_days, "original_estimated_days": original_estimated_days,
        }))
        model_records.append({"model": name, "suffix": suffix, "loss": loss_kind,
                              "center": center, "under_weight": under_w,
                              "state_dict": model_path.name})

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_predictions.to_csv(OUTPUT_DIR / f"{PREFIX}model_predictions.csv",
                           index=False, encoding="utf-8-sig")

    # ---- 元信息：val / ooc 复现同样的序列与标准化 ----
    meta = {
        "cutoff_time": cutoff_time,
        "full_start_time": FULL_START_TIME,
        "cat_categories": cat_categories,
        "seq_stats": seq_stats, "static_stats": static_stats,
        "snap_mean": snap_mean, "seq_t0": seq_t0,
        "seq_steps": CONFIG["tc_seq_steps"], "bucket_days": CONFIG["tc_bucket_days"],
        "seq_mode": CONFIG["tc_seq_mode"], "snap_dim": SNAP_DIM,
        "numeric_cols": [x for x in data.columns
                         if x not in EXCLUDE_COLS + CAT_FEATURES
                         and pd.api.types.is_numeric_dtype(data[x])],
        "model_records": model_records,
    }
    joblib.dump(meta, OUTPUT_DIR / f"{PREFIX}meta.joblib")
    print(f"元信息已保存：{OUTPUT_DIR / (PREFIX + 'meta.joblib')}")

    # ---- 评估与图（结构同 lgb_model.py 第 7~11 节）----
    metrics_df = evaluate_predictions(all_predictions, "TransformerVCN 测试集")
    metrics_df.to_csv(OUTPUT_DIR / f"{PREFIX}timeout_model_comparison.csv",
                      index=False, encoding="utf-8-sig")

    save_distribution_table(all_predictions, "TransformerVCN 测试集",
                            f"{PREFIX}delivery_time_distribution.csv")
    plot_distribution(all_predictions, "TransformerVCN 测试集",
                      f"{PREFIX}delivery_time_distribution.svg")
    plot_error_distribution(all_predictions, "TransformerVCN 测试集",
                            f"{PREFIX}delivery_time_error_distribution.svg")
    plot_delay_rate(metrics_df, "TransformerVCN 测试集",
                    f"{PREFIX}delivery_delay_rate_comparison.svg")

    summary = save_summary(metrics_df, "TransformerVCN 测试集",
                           f"{PREFIX}mean_duration_and_risk_comparison.csv")
    print("\n========== 预测精度、提前送达率与延迟风险对比 ==========")
    print(summary.round(3).to_string(index=False))
    print(f"\n表格保存目录：{OUTPUT_DIR.resolve()}\n图表保存目录：{ASSET_DIR.resolve()}")
